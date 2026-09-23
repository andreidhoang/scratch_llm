"""core/kda.py — Kimi Delta Attention (K2 rung): the operator, two ways, and the K3 layer.

State convention (keys on rows, values on columns), per value head:
    D_t = Diag(exp(g_t))                                   # one decay per KEY channel
    S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T (scale * q_t)                              # reads the UPDATED state

Operator contract (fla/ops/kda/naive.py layout, so one set of inputs feeds both):
    q, k  [B, T, H, K]   L2-normalised       v     [B, T, HV, V]   HV % H == 0 (GVA)
    g     [B, T, HV, K]  log-decay, <= 0     beta  [B, T, HV]      in [0, 1]
    S     [B, HV, K, V]  fp32, or fp64 when any input is fp64; o comes back in v.dtype
K3 itself has HV == H.

Layer (docs/k3/KDA_ALOG_MAPPING.md, docs/k3/K2_PROPOSAL_KDA.md):
* Per-channel decay is Kimi Linear's KDA. K3 changes the decay map to
  g = lower_bound * sigmoid(exp(A_log[h]) * (z + dt_bias)), lower_bound = -5, and makes the
  output gate full-rank.
* A_log is per head. The checkpoint stores [128] = 96 live entries + a zero tail; the loader
  takes either length and refuses a nonzero tail.
* Q, K, V have independent causal-conv histories; the cache holds PRE-conv inputs.
* Output is o_proj(RMSNorm(o) * sigmoid(g_proj(x))): norm first, then gate.
* FLA and the K3 reference keep the state value-first ([V, K]); transpose before comparing
  with, or loading, an external cache.

Chunkwise is a real unit-lower-triangular solve per chunk, written for correctness, not speed.
Autograd still keeps activations; only the inference cache is O(1) in sequence length.

Not implemented: masks, packed sequences, the Kimi Linear softplus map and low-rank gate,
quantized checkpoints, beam reordering. One batch row is one continuous stream; reset its
KDAState when the stream changes. Stateful forward detaches the cache (no cross-call BPTT).

Sources (reviewed 2026-09-22): arXiv:2510.26692; huggingface.co/moonshotai/Kimi-K3
modeling_kimi_linear.py; fla/ops/kda/{naive,gate}.py and fla/layers/kda.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_llm.k3.config import KDAConfig

if TYPE_CHECKING:
    from scratch_llm.k3.model import HybridState

__all__ = ["KDAState", "KDALayer", "kda_recurrent", "kda_chunk"]


# ---------------------------------------------------------------------------
# Operator — pure functions, no parameters (tests/test_kda_parity_fla.py).
# ---------------------------------------------------------------------------


def _math_dtype(*tensors: Tensor | None) -> torch.dtype:
    """fp64 if any input is fp64, else fp32: the state never accumulates in half precision."""
    if any(t is not None and t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


class _Core(NamedTuple):
    q: Tensor  # [B, T, HV, K], heads expanded, scaled
    k: Tensor  # [B, T, HV, K], heads expanded
    v: Tensor  # [B, T, HV, V]
    g: Tensor  # [B, T, HV, K]
    beta: Tensor  # [B, T, HV]
    state: Tensor  # [B, HV, K, V], never aliases the caller's initial_state


def _prepare(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float | None,
    initial_state: Tensor | None,
    check_ranges: bool,
) -> _Core:
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q must be [B,T,H,K], v [B,T,HV,V]; got {q.shape}, {v.shape}")
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if k.shape != q.shape or v.shape[:2] != (B, T) or HV % H != 0:
        raise ValueError(f"k must match q, v must be [B,T,HV,V] with HV % H == 0; got {v.shape}")
    if g.shape != (B, T, HV, K) or beta.shape != (B, T, HV):
        raise ValueError(f"g must be {(B, T, HV, K)} and beta {(B, T, HV)}")
    if initial_state is not None and initial_state.shape != (B, HV, K, V):
        raise ValueError(f"initial_state must be {(B, HV, K, V)}; got {initial_state.shape}")
    scale = 1.0 / math.sqrt(K) if scale is None else scale

    dtype = _math_dtype(q, k, v, g, beta, initial_state)
    q, k, v, g, beta = (t.to(dtype) for t in (q, k, v, g, beta))
    # Host sync; KDALayer skips it because its gates are in range by construction.
    if check_ranges:
        if not bool((torch.isfinite(g) & (g <= 0)).all()):
            raise ValueError("g (log-decay) must be finite and <= 0")
        if not bool(((beta >= 0) & (beta <= 1)).all()):
            raise ValueError("beta must lie in [0, 1]")
    q = q.repeat_interleave(HV // H, dim=2) * scale
    k = k.repeat_interleave(HV // H, dim=2)
    if initial_state is None:
        state = q.new_zeros(B, HV, K, V)
    else:
        state = initial_state.to(dtype, copy=True)
    return _Core(q, k, v, g, beta, state)


def kda_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    *,
    check_ranges: bool = True,
) -> tuple[Tensor, Tensor]:
    """Token-by-token scan: the definition of the operator, and the decode path.

    Per step: decay, then the delta-rule erase/write against the DECAYED state, then the read.
    Returns (o [B, T, HV, V], final state [B, HV, K, V]); inputs are never mutated.
    """
    with torch.autocast(q.device.type, enabled=False):  # keep the state out of bf16 matmuls
        c = _prepare(q, k, v, g, beta, scale, initial_state, check_ranges)
        state, outputs = c.state, []
        for t in range(c.q.shape[1]):
            k_t = c.k[:, t]  # [B, HV, K]
            state = state * c.g[:, t, :, :, None].exp()  # D_t S: scales rows (key channels)
            pred = torch.einsum("bhk,bhkv->bhv", k_t, state)  # k_t^T D_t S
            write = c.beta[:, t, :, None] * (c.v[:, t] - pred)  # beta_t (v_t - k_t^T D_t S)
            state = state + k_t[..., None] * write[..., None, :]
            outputs.append(torch.einsum("bhk,bhkv->bhv", c.q[:, t], state))
    o = torch.stack(outputs, dim=1) if outputs else c.v[:, :0]
    return o.to(v.dtype), state


def _intra_chunk_scores(
    q: Tensor, k: Tensor, G: Tensor, channel_tile: int
) -> tuple[Tensor, Tensor]:
    """A_kk[i, j] = <k_i, exp(G_i - G_j) k_j> and A_qk[i, j] = <q_i, exp(G_i - G_j) k_j> for
    j <= i, zero above the diagonal. Inputs [B, HV, C, K]; outputs [B, HV, C, C].

    The decay sits inside the dot product (one factor per channel), so it does not factor out
    as a scalar. A future pair (j > i) would need exp(G_i - G_j) > 1, which overflows fp32 once
    a chunk's cumulative decay passes ~88 (g ~ -1.4 per step over 64 tokens) and turns into
    0 * inf = NaN in the backward even though the forward masks it. Future pairs are therefore
    masked to -inf BEFORE the exp.
    """
    C = G.shape[-2]
    future = torch.ones(C, C, dtype=torch.bool, device=G.device).triu(1)[..., None]  # [C, C, 1]
    A_kk = G.new_zeros(*G.shape[:-2], C, C)
    A_qk = torch.zeros_like(A_kk)
    for d in range(0, G.shape[-1], channel_tile):  # bounds the [.., C, C, tile] temporary
        cols = slice(d, d + channel_tile)
        Gd, kd, qd = G[..., cols], k[..., cols], q[..., cols]
        decay = (Gd[..., :, None, :] - Gd[..., None, :, :]).masked_fill(future, -math.inf).exp()
        decayed_k = decay * kd[..., None, :, :]  # [.., i, j, c] = exp(G_i - G_j)_c k_j,c
        A_kk = A_kk + (kd[..., :, None, :] * decayed_k).sum(-1)
        A_qk = A_qk + (qd[..., :, None, :] * decayed_k).sum(-1)
    return A_kk, A_qk


def _chunk_step(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, state: Tensor, channel_tile: int
) -> tuple[Tensor, Tensor]:
    """One chunk. q (scaled), k, g: [B, HV, C, K]; v: [B, HV, C, V]; beta: [B, HV, C, 1];
    state: [B, HV, K, V] entering the chunk. Returns (o [B, HV, C, V], state leaving it).

    With G = cumsum(g) and P = exp(G), unrolling the recurrence inside the chunk makes the
    write vectors rho_i = beta_i (v_i - k_i^T D_i S_{i-1}) the solution of one unit-lower-
    triangular system:
        (I + L) rho = beta v - (beta k P) S,   L_ij = beta_i A_kk[i, j] for i > j
        rho = U - W S,   U = (I + L)^-1 beta v,   W = (I + L)^-1 beta k P
    then o = (q P) S + A_qk rho and S_out = P_C S + (k exp(G_C - G))^T rho.
    """
    G = g.cumsum(dim=-2)
    P = G.exp()
    A_kk, A_qk = _intra_chunk_scores(q, k, G, channel_tile)
    C, K, V = G.shape[-2], k.shape[-1], v.shape[-1]
    system = (beta * A_kk).tril(-1) + torch.eye(C, dtype=G.dtype, device=G.device)
    rhs = torch.cat((beta * v, beta * k * P), dim=-1)  # both right-hand sides, one solve
    solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    U, W = solved.split((V, K), dim=-1)
    rho = U - W @ state
    o = (q * P) @ state + A_qk @ rho
    to_end = k * (G[..., -1:, :] - G).exp()  # each write decayed to the end of the chunk
    return o, P[..., -1, :, None] * state + to_end.transpose(-2, -1) @ rho


def kda_chunk(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    chunk_size: int = 64,
    *,
    channel_tile: int = 16,
    check_ranges: bool = True,
) -> tuple[Tensor, Tensor]:
    """Chunkwise-parallel form, the training path: same contract and function as
    :func:`kda_recurrent`. The last chunk may be shorter than ``chunk_size``."""
    if chunk_size < 1 or channel_tile < 1:
        raise ValueError(
            f"chunk_size and channel_tile must be >= 1; got {chunk_size}, {channel_tile}"
        )
    with torch.autocast(q.device.type, enabled=False):
        c = _prepare(q, k, v, g, beta, scale, initial_state, check_ranges)
        state, outputs = c.state, []
        for start in range(0, c.q.shape[1], chunk_size):
            span = slice(start, start + chunk_size)
            qc, kc, vc, gc = (t[:, span].transpose(1, 2) for t in (c.q, c.k, c.v, c.g))
            bc = c.beta[:, span].transpose(1, 2)[..., None]
            o, state = _chunk_step(qc, kc, vc, gc, bc, state, channel_tile)
            outputs.append(o.transpose(1, 2))
    out = torch.cat(outputs, dim=1) if outputs else c.v[:, :0]
    return out.to(v.dtype), state


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


@dataclass
class KDAState:
    """Decode cache for one KDA layer, all batch rows; constant size in sequence length.

    s_t:  [B, H, D, D]  delta-rule state, keys on rows; fp32 (fp64 for fp64 activations).
    conv: [B, 3, H*D, kernel - 1]  raw (pre-conv) Q/K/V projections, oldest first.
    Zeros are exact: a zero state and a zero history are what prefill assumes at t = 0.
    """

    s_t: Tensor
    conv: Tensor

    @classmethod
    def zeros(
        cls, cfg: KDAConfig, batch: int, device: torch.device, dtype: torch.dtype
    ) -> KDAState:
        H, D = cfg.num_heads, cfg.head_dim
        state_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        return cls(
            s_t=torch.zeros(batch, H, D, D, device=device, dtype=state_dtype),
            conv=torch.zeros(batch, 3, H * D, cfg.conv_kernel_size - 1, device=device, dtype=dtype),
        )


class _ShortConv(nn.Module):
    """Depthwise causal conv + SiLU over [B, T, P], continuing from a raw-input history.

    Weight [P, 1, kernel] with nn.Conv1d's init and no bias, as fla's ShortConvolution, so
    checkpoint tensors load unchanged. Cross-correlation: the last tap sees the current token.
    """

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Conv1d.reset_parameters

    def forward(self, x: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        """x [B, T, P], history [B, P, kernel - 1] -> (SiLU(conv(x)) [B, T, P], new history)."""
        raw = torch.cat((history.to(x.dtype), x.transpose(1, 2)), dim=-1)  # [B, P, L + T]
        dtype = _math_dtype(x, self.weight)
        with torch.autocast(x.device.type, enabled=False):
            y = F.silu(F.conv1d(raw.to(dtype), self.weight.to(dtype), groups=raw.shape[1]))
        keep = self.weight.shape[-1] - 1
        return y.transpose(1, 2).to(x.dtype), raw[..., raw.shape[-1] - keep :].clone()


def _l2_normalize(x: Tensor, eps: float) -> Tensor:
    """x * rsqrt(sum(x^2) + eps), fla's l2norm (not x / max(|x|, eps))."""
    with torch.autocast(x.device.type, enabled=False):
        z = x.to(_math_dtype(x))
        return (z * torch.rsqrt(z.square().sum(-1, keepdim=True) + eps)).to(x.dtype)


class _RMSNormGated(nn.Module):
    """RMSNorm over head_dim (weight shared across heads), then * sigmoid(gate)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        with torch.autocast(x.device.type, enabled=False):
            dtype = _math_dtype(x, gate, self.weight)
            z = x.to(dtype)
            z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
            return (z * self.weight.to(dtype) * gate.to(dtype).sigmoid()).to(x.dtype)


class KDALayer(nn.Module):
    """One K3 KDA layer: forward(x [B, T, hidden], state=None, layer_id=0) -> [B, T, hidden].

        q, k, v = SiLU(causal_conv(W x)); q, k L2-normalised
        g       = lower_bound * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias))
        beta    = sigmoid(b_proj(x))
        out     = o_proj(RMSNorm(kda(q, k, v, g, beta)) * sigmoid(g_proj(x)))

    With ``state=None`` (training prefill) the operator runs chunkwise from a zero state. With a
    HybridState the layer reads and writes ``state.kda_states[layer_id - 1]``: one token takes
    the recurrent step, several take the chunkwise path from the cached state.
    """

    def __init__(
        self,
        cfg: KDAConfig,
        hidden_size: int,
        *,
        chunk_size: int = 64,
        norm_eps: float = 1e-5,
        l2_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not cfg.full_rank_gate:
            raise NotImplementedError("only K3's full-rank output gate is implemented")
        if cfg.a_log_size < cfg.num_heads:
            raise ValueError("a_log_size is checkpoint storage and must be >= num_heads")
        if not cfg.gate_lower_bound < 0:
            raise ValueError("gate_lower_bound must be negative")
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.l2_eps = l2_eps
        H, D, P = cfg.num_heads, cfg.head_dim, cfg.projection_size

        self.q_proj = nn.Linear(hidden_size, P, bias=False)
        self.k_proj = nn.Linear(hidden_size, P, bias=False)
        self.v_proj = nn.Linear(hidden_size, P, bias=False)
        self.q_conv1d = _ShortConv(P, cfg.conv_kernel_size)
        self.k_conv1d = _ShortConv(P, cfg.conv_kernel_size)
        self.v_conv1d = _ShortConv(P, cfg.conv_kernel_size)
        self.f_a_proj = nn.Linear(hidden_size, cfg.decay_rank, bias=False)
        self.f_b_proj = nn.Linear(cfg.decay_rank, P, bias=False)
        self.b_proj = nn.Linear(hidden_size, H, bias=False)
        # Gate init as fla's safe_gate path (K2_PROPOSAL_KDA §2): exp(A_log) = 1, and dt_bias is
        # softplus^-1 of dt ~ logU(1e-3, 1e-1), so at z = 0 the log-decay is ~ -5 * sigmoid(-4.6).
        self.A_log = nn.Parameter(torch.zeros(H))
        dt = torch.empty(P).uniform_(math.log(1e-3), math.log(1e-1)).exp().clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.g_proj = nn.Linear(hidden_size, P, bias=False)  # full-rank output gate
        self.o_norm = _RMSNormGated(D, norm_eps)
        self.o_proj = nn.Linear(P, hidden_size, bias=False)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Accept the checkpoint's padded A_log [a_log_size]: keep [0:num_heads], refuse a
        nonzero tail rather than guess what it means."""
        key, H = prefix + "A_log", self.cfg.num_heads
        a = state_dict.get(key)
        if a is not None and a.ndim == 1 and a.numel() == self.cfg.a_log_size != H:
            if torch.count_nonzero(a[H:]) > 0:
                error_msgs.append(f"{key}: nonzero padding beyond num_heads={H}; refusing to load")
            else:
                state_dict = {**state_dict, key: a[:H]}  # never rewrite the caller's dict
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _read_cache(self, state: HybridState, layer_id: int, x: Tensor) -> KDAState:
        if layer_id < 1:
            raise ValueError("stateful forward needs the 1-based layer_id")
        cache = state.kda_states[layer_id - 1]
        B = x.shape[0]
        if cache is None:
            return KDAState.zeros(self.cfg, B, x.device, x.dtype)
        H, D, L = self.cfg.num_heads, self.cfg.head_dim, self.cfg.conv_kernel_size - 1
        if cache.s_t.shape != (B, H, D, D) or cache.conv.shape != (B, 3, H * D, L):
            raise ValueError("KDAState does not match this layer and batch; reset the cache")
        return cache

    def _qkv(self, x: Tensor, cache: KDAState | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project, causal short conv, SiLU. Returns q, k, v [B, T, H, D] and the new raw
        histories [B, 3, H*D, kernel - 1]."""
        B, T, _ = x.shape
        H, D = self.cfg.num_heads, self.cfg.head_dim
        L = self.cfg.conv_kernel_size - 1
        outs, histories = [], []
        for i, (proj, conv) in enumerate(
            (
                (self.q_proj, self.q_conv1d),
                (self.k_proj, self.k_conv1d),
                (self.v_proj, self.v_conv1d),
            )
        ):
            history = x.new_zeros(B, H * D, L) if cache is None else cache.conv[:, i].detach()
            y, history = conv(proj(x), history)
            outs.append(y.view(B, T, H, D))
            histories.append(history)
        return outs[0], outs[1], outs[2], torch.stack(histories, dim=1)

    def _gates(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """log-decay g [B, T, H, D] in [lower_bound, 0] and beta [B, T, H] in [0, 1]."""
        B, T, _ = x.shape
        H, D = self.cfg.num_heads, self.cfg.head_dim
        z = self.f_b_proj(self.f_a_proj(x)).view(B, T, H, D)
        b = self.b_proj(x)
        with torch.autocast(x.device.type, enabled=False):
            dtype = _math_dtype(z, self.dt_bias)
            z = z.to(dtype) + self.dt_bias.to(dtype).view(H, D)
            scale = self.A_log.to(dtype).exp()[:, None]  # one pre-scale per head
            g = self.cfg.gate_lower_bound * torch.sigmoid(scale * z)
            beta = torch.sigmoid(b.to(dtype))
        return g, beta

    def forward(
        self,
        x: Tensor,  # (B, T, hidden) — already RMSNorm-ed by the block
        state: HybridState | None = None,
        layer_id: int = 0,
    ) -> Tensor:
        B, T, _ = x.shape
        H, D = self.cfg.num_heads, self.cfg.head_dim
        cache = None if state is None else self._read_cache(state, layer_id, x)

        q, k, v, conv = self._qkv(x, cache)
        q, k = _l2_normalize(q, self.l2_eps), _l2_normalize(k, self.l2_eps)
        g, beta = self._gates(x)
        s0 = None if cache is None else cache.s_t.detach()
        if T == 1 and s0 is not None:  # decode step
            o, s_t = kda_recurrent(q, k, v, g, beta, initial_state=s0, check_ranges=False)
        else:  # training prefill, or a multi-token extend from the cache
            o, s_t = kda_chunk(
                q, k, v, g, beta, initial_state=s0, chunk_size=self.chunk_size, check_ranges=False
            )

        out = self.o_proj(self.o_norm(o, self.g_proj(x).view(B, T, H, D)).reshape(B, T, H * D))
        if state is not None:  # commit only after the whole forward succeeded
            state.kda_states[layer_id - 1] = KDAState(s_t.detach(), conv.detach())
        return out
