"""Gated-DeltaNet linear-attention block (F10.1) — the 2026 hybrid-attention frontier.

Intent
------
A from-scratch Gated DeltaNet block (Yang et al., "Gated Delta Networks", arXiv:2412.06464 —
the linear-attention layer of Qwen3-Next and the ancestor of Kimi-Linear's KDA), implemented
as THREE forward paths over one recurrence:

    S_t = (I − β_t k_t k_tᵀ) · (α_t S_{t-1}) + β_t k_t v_tᵀ        y_t = S_tᵀ q_t     (per head)

with S ∈ R^{d_k × d_v} the fast-weight state, α_t ∈ (0,1) a per-head data-dependent decay and
β_t ∈ (0,1) the delta-rule write strength. (The taskspec's ``S_{t-1}·(diag(α_t) − β_t k_t k_tᵀ)``
shorthand is realized in the Yang-et-al./Qwen3-Next convention: the scalar per-head decay is
applied *before* the rank-1 delta projection reads the state — this matches the vendored
``transformers/models/qwen3_next`` torch reference, our oracle.)

Invariant (the whole point of the rung)
---------------------------------------
``chunkwise == recurrent-scan == float64 reference``: the chunkwise-parallel training path,
the single-step decode path, and the naive reference loop must compute the SAME function
(tests/test_linear_attn.py). The same math must serve train and decode — any drift between
the paths is train/serve skew, and the DELTA GDN-2 kernel that later accelerates ``step``
inherits its correctness contract from this equivalence.

Interview question
------------------
Why did 2026 labs (Qwen3-Next, Kimi-Linear, MiniMax) move to linear-attention hybrids; what
does the delta rule's rank-1 state update buy over vanilla linear attention; and why is the
low-precision state-materializing decode kernel the hard part?
— Hybrids amortize the O(T) KV-cache memory wall: a constant-size state replaces a growing
  cache on 3 of every 4 layers, so long-context decode stays memory-bound on weights, not KV.
— Vanilla linear attention (S_t = S_{t-1} + k v ᵀ) only ever *adds*; the delta rule first
  *erases* the old association along k (the −β k kᵀ S term is an online rank-1 least-squares
  correction), giving bounded state norm and usable associative recall.
— The decode kernel is hard because S (d_k × d_v per head) must be *materialized and updated
  in registers/SMEM every step* under low precision: the recurrence is a long product of
  contractions, so fp16/bf16 state accumulation drifts; the kernel must pick where fp32
  lives (state) and where bf16 suffices (q/k/v) — exactly the DELTA GDN-2 spike.

Conventions (mirroring the Qwen3-Next torch reference, re-owned not copied)
---------------------------------------------------------------------------
- q/k are L2-normalized per head and q is scaled by d_k^{-1/2} (delta-rule stability: with
  ‖k‖=1 and β ∈ (0,1), I − β k kᵀ is a contraction — Yang et al. 2412.06464).
- α_t = exp(−exp(A_log) · softplus(a_t + dt_bias)) with per-head learned A_log, dt_bias and
  a data-dependent projection a_t — the Qwen3-Next/GDN parameterization (Mamba-2 lineage:
  log-space decay keeps α ≈ 1 reachable with well-scaled gradients, vs a plain sigmoid).
- β_t = sigmoid(b_t); value expansion d_v = expand_v · d_head (GDN convention, expand_v=2).
- Scope deviations at this rung (F10.1): no short causal conv1d pre-QKV and no z-gated
  output RMSNorm (Qwen3-Next block extras — quality add-ons orthogonal to the recurrence
  contract; they join in F10.2 where the block competes on quality). No model wiring here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "GatedDeltaNet",
    "LinearAttnConfig",
    "LinearAttnState",
    "gated_delta_rule_chunkwise",
    "gated_delta_rule_reference",
    "gated_delta_rule_step",
    "linear_attn_state_bytes",
]


@dataclass(frozen=True)
class LinearAttnConfig:
    """Gated-DeltaNet hyperparameters.

    ``d_v = expand_v · d_head`` (GDN value expansion); the block's model width is
    ``d_model = n_heads · d_head`` (the F10.2 wiring maps the host model's d_model here).
    """

    d_head: int
    n_heads: int
    expand_v: int = 2
    chunk_size: int = 64

    @property
    def d_v(self) -> int:
        return self.expand_v * self.d_head

    @property
    def d_model(self) -> int:
        return self.n_heads * self.d_head


@dataclass
class LinearAttnState:
    """Constant-size recurrent state — the object that replaces a growing KV cache.

    ``S``: [B, n_heads, d_k, d_v] fast-weight matrix per head. That is ALL the state the
    recurrence carries (no conv state at this rung), so decode memory is constant in T —
    the memory-wall claim tested against a GQA-8 KV cache in tests/test_linear_attn.py.
    """

    S: Tensor

    def bytes(self) -> int:
        return self.S.numel() * self.S.element_size()


def linear_attn_state_bytes(cfg: LinearAttnConfig, dtype: torch.dtype) -> int:
    """Analytic per-sequence (B=1) state size: n_heads · d_head · d_v · itemsize.

    Constant in T by construction — compare with a KV cache's 2 · n_kv · d_head · T · itemsize.
    """
    return cfg.n_heads * cfg.d_head * cfg.d_v * dtype.itemsize


# ---------------------------------------------------------------------------
# Path 3 — recurrent-scan single step (the decode contract DELTA accelerates)
# ---------------------------------------------------------------------------


def gated_delta_rule_step(
    state: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    alpha_t: Tensor,
    beta_t: Tensor,
) -> tuple[Tensor, Tensor]:
    """One token of the gated delta rule, batched over [B, H].

    state: [B, H, d_k, d_v] · q_t/k_t: [B, H, d_k] · v_t: [B, H, d_v] · alpha_t/beta_t: [B, H].
    Returns (y_t [B, H, d_v], new_state). Functional — the caller owns the state (the fused
    decode kernel will update in place; the contract here is the *math*, not the storage).
    """
    decayed = state * alpha_t[..., None, None]  # α_t S_{t-1}
    kv_mem = (decayed * k_t[..., :, None]).sum(dim=-2)  # (α_t S)ᵀ k_t — read after decay
    delta = (v_t - kv_mem) * beta_t[..., None]  # β_t (v_t − prediction)
    new_state = decayed + k_t[..., :, None] * delta[..., None, :]  # rank-1 write along k_t
    y_t = (new_state * q_t[..., :, None]).sum(dim=-2)  # S_tᵀ q_t
    return y_t, new_state


# ---------------------------------------------------------------------------
# Path 2 — chunkwise-parallel forward (the training path)
# ---------------------------------------------------------------------------


def _chunk_forward(
    state: Tensor,
    q_c: Tensor,
    k_c: Tensor,
    v_c: Tensor,
    alpha_c: Tensor,
    beta_c: Tensor,
) -> tuple[Tensor, Tensor]:
    """Process ONE chunk given the entering state; return (y_chunk, leaving state).

    This signature IS the chunk boundary: state enters and leaves only here. At this rung
    the intra-chunk computation is a per-token loop (taskspec concession); F10.2/DELTA
    replace this body with the chunkwise WY/UT-decomposition matmuls without touching the
    chunk→chunk state-passing structure.
    """
    ys = []
    for t in range(q_c.shape[2]):
        y_t, state = gated_delta_rule_step(
            state, q_c[:, :, t], k_c[:, :, t], v_c[:, :, t], alpha_c[:, :, t], beta_c[:, :, t]
        )
        ys.append(y_t)
    return torch.stack(ys, dim=2), state


def gated_delta_rule_chunkwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    chunk_size: int,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Chunkwise-parallel gated delta rule: y [B, H, T, d_v] and the final state.

    q/k: [B, H, T, d_k] · v: [B, H, T, d_v] · alpha/beta: [B, H, T]. Ragged tails are fine
    (the last chunk is simply shorter); ``chunk_size > T`` degenerates to one chunk. The
    output must be independent of chunk_size — tested as bitwise equality.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    bsz, n_heads, seq_len, d_k = q.shape
    d_v = v.shape[-1]
    state = (
        torch.zeros(bsz, n_heads, d_k, d_v, dtype=q.dtype, device=q.device)
        if initial_state is None
        else initial_state
    )
    y_chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        y_c, state = _chunk_forward(
            state,
            q[:, :, start:end],
            k[:, :, start:end],
            v[:, :, start:end],
            alpha[:, :, start:end],
            beta[:, :, start:end],
        )
        y_chunks.append(y_c)
    return torch.cat(y_chunks, dim=2), state


# ---------------------------------------------------------------------------
# Path 1 — float64 reference loop (the oracle, written for clarity)
# ---------------------------------------------------------------------------


def gated_delta_rule_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor]:
    """Naive per-token recurrence with EXPLICIT transition matrices, in float64.

    Deliberately different algebra from the fused step (builds I − β k kᵀ and matmuls it)
    so the equivalence test has teeth. Per head:

        S_t = (I − β_t k_t k_tᵀ) @ (α_t S_{t-1}) + β_t k_t v_tᵀ ;   y_t = S_tᵀ q_t
    """
    q, k, v, alpha, beta = (x.to(torch.float64) for x in (q, k, v, alpha, beta))
    bsz, n_heads, seq_len, d_k = q.shape
    d_v = v.shape[-1]
    y = torch.zeros(bsz, n_heads, seq_len, d_v, dtype=torch.float64)
    s_final = torch.zeros(bsz, n_heads, d_k, d_v, dtype=torch.float64)
    eye = torch.eye(d_k, dtype=torch.float64)
    for b in range(bsz):
        for h in range(n_heads):
            s = torch.zeros(d_k, d_v, dtype=torch.float64)
            for t in range(seq_len):
                k_t = k[b, h, t]  # [d_k]
                erase = eye - beta[b, h, t] * torch.outer(k_t, k_t)  # I − β k kᵀ
                write = beta[b, h, t] * torch.outer(k_t, v[b, h, t])  # β k vᵀ
                s = erase @ (alpha[b, h, t] * s) + write
                y[b, h, t] = s.T @ q[b, h, t]
            s_final[b, h] = s
    return y, s_final


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """FLA/Qwen3-Next l2norm: x · rsqrt(Σx² + eps) (eps inside the sqrt, on the square-sum)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class GatedDeltaNet(nn.Module):
    """Gated-DeltaNet block: project → gate → gated delta rule (chunkwise) → out-project.

    forward() is the chunkwise training path; ``project`` + ``step`` form the decode path
    — both compute the same function (the module invariant). d_model = n_heads · d_head.
    """

    def __init__(self, cfg: LinearAttnConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_model, n_heads = cfg.d_model, cfg.n_heads
        self.q_proj = nn.Linear(d_model, n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * cfg.d_v, bias=False)
        # Per-head gates: a_proj → decay logits (α), b_proj → write-strength logits (β).
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        # Qwen3-Next/GDN decay parameterization: α = exp(−exp(A_log)·softplus(a + dt_bias)).
        # A ~ U(1, 16) (Mamba-lineage init; lower bound 1 avoids log(0) vs Qwen's U(0, 16)).
        self.A_log = nn.Parameter(torch.log(torch.empty(n_heads).uniform_(1.0, 16.0)))
        self.dt_bias = nn.Parameter(torch.ones(n_heads))
        self.out_proj = nn.Linear(n_heads * cfg.d_v, d_model, bias=False)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """x [B, T, d_model] → (q, k, v, alpha, beta) in the [B, H, T, *] convention.

        q/k L2-normalized per head, q scaled by d_k^{-1/2} (both linear in the recurrence,
        so applied once here and shared verbatim by the chunkwise and decode paths).
        Gates are computed in ≥ float32 (fp16-safe: exp/softplus over learned magnitudes),
        then cast back to the activation dtype.
        """
        bsz, seq_len, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(bsz, seq_len, cfg.n_heads, cfg.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, cfg.n_heads, cfg.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, cfg.n_heads, cfg.d_v).transpose(1, 2)
        q = _l2norm(q) * (1.0 / math.sqrt(cfg.d_head))
        k = _l2norm(k)

        gate_dtype = torch.promote_types(x.dtype, torch.float32)
        a = self.a_proj(x).to(gate_dtype).transpose(1, 2)  # [B, H, T]
        b = self.b_proj(x).to(gate_dtype).transpose(1, 2)
        decay_rate = torch.exp(self.A_log.to(gate_dtype))[None, :, None]  # A > 0, per head
        log_alpha = -decay_rate * F.softplus(a + self.dt_bias.to(gate_dtype)[None, :, None])
        alpha = torch.exp(log_alpha).to(x.dtype)  # ∈ (0, 1)
        beta = torch.sigmoid(b).to(x.dtype)  # ∈ (0, 1)
        return q, k, v, alpha, beta

    def forward(self, x: Tensor, chunk_size: int | None = None) -> Tensor:
        """Chunkwise training path: [B, T, d_model] → [B, T, d_model]."""
        q, k, v, alpha, beta = self.project(x)
        y, _ = gated_delta_rule_chunkwise(
            q, k, v, alpha, beta, chunk_size=chunk_size or self.cfg.chunk_size
        )
        bsz, _, seq_len, _ = y.shape
        y = y.transpose(1, 2).reshape(bsz, seq_len, self.cfg.n_heads * self.cfg.d_v)
        return self.out_proj(y)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LinearAttnState:
        cfg = self.cfg
        return LinearAttnState(
            S=torch.zeros(batch_size, cfg.n_heads, cfg.d_head, cfg.d_v, device=device, dtype=dtype)
        )

    def step(
        self,
        state: LinearAttnState,
        q_t: Tensor,
        k_t: Tensor,
        v_t: Tensor,
        alpha_t: Tensor,
        beta_t: Tensor,
    ) -> tuple[Tensor, LinearAttnState]:
        """Single-token decode: the recurrent contract the DELTA GDN-2 kernel accelerates.

        Inputs are one time-slice of ``project`` outputs ([B, H, d] / [B, H]); returns the
        per-head core output y_t [B, H, d_v] (caller merges heads + applies out_proj) and
        the successor state. Must be token-exact vs the chunkwise path (tested).
        """
        y_t, new_s = gated_delta_rule_step(state.S, q_t, k_t, v_t, alpha_t, beta_t)
        return y_t, LinearAttnState(S=new_s)
