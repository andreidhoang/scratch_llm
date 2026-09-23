"""core/gated_mla.py — Gated Multi-head Latent Attention (K3 rung): one layer, two exact paths.

Per token, h heads, per-head dims nope | rope | v (full scale 128 | 64 | 128, kv_lora 512):
    q          = q_b_proj(RMSNorm(q_a_proj x))       viewed [h, nope + rope] = [q_nope | q_rope]
    c, k_rope  = split(kv_a_proj_with_mqa x, [kv_lora, rope]);   c = RMSNorm(c)
    k_nope, v  = split(kv_b_proj c, [nope, v])        per head
    logit_hts  = (q_nope·k_nope_s + q_rope·k_rope_s) / sqrt(nope + rope),     causal: s <= t
    out        = o_proj( softmax_s(logit) v  ⊙  sigmoid(g_proj x) )

* No RoPE anywhere (config ``mla_use_nope``). "rope" is DeepSeek heritage only: ``k_rope`` is
  ONE unrotated, un-normed key part shared by every head, and it is live — 64 of the 192 score
  dims at full scale (32 of 96 in ``mini_k3_d12``), hence the scale (nope + rope)^-1/2.
* The gate reads the layer input x (the block's normed sublayer input), has width h·v, and sits
  BEFORE o_proj (arXiv:2505.06708; HF :470-473). A gate after o_proj would be H -> H.
* The decode cache is the latent only, :class:`MLALatentKV` = (normed c, raw k_rope): per token
  kv_lora + rope values, independent of the head count.

Absorption. W = kv_b_proj.weight viewed [h, nope + v, kv_lora] splits into per-head row blocks
W_UK [h, nope, kv_lora] and W_UV [h, v, kv_lora], with k_nope_s = W_UK c_s and v_s = W_UV c_s:
    q_nope·k_nope_s = (W_UK^T q_nope)·c_s        sum_s p_s v_s = W_UV (sum_s p_s c_s)
so attention can run in the latent space (score dim kv_lora + rope) without ever building
per-head K/V. :meth:`GatedMLA.attend_naive` materializes K/V (the HF formulation);
:meth:`GatedMLA.attend_absorbed` folds W_UK into the query and applies W_UV after the latent sum
(vLLM's decode). Same function: they agree in fp64 to rounding.

Dispatch. Training prefill (no state) takes the naive path: T queries over the same T keys cost
less in the nope + rope score space (192 dims) than in the latent one (576). Any call with a cache
(decode or a multi-token extend) takes the absorbed path, so the cache is never expanded to
per-head K/V. The cache grows by concatenation, an O(S) copy per call — a reference, not a paged
cache. Stateful forward detaches the cache (no cross-call BPTT).

Precision: projections run in the module dtype; scores, softmax and both weighted sums run in
fp32 (fp64 for fp64 activations) with autocast off; the gated output is cast back before o_proj.
The two latent norms use HF's KimiRMSNorm default eps 1e-6 (vLLM passes ``rms_norm_eps`` = 1e-5
to both).

Row layout for per-head Muon and QK-Clip (k3/muon.py): q_b_proj rows are [h, nope + rope], nope
first; kv_b_proj rows are [h, nope + v], k_nope first. ``track_max_logits`` feeds QK-Clip.

Not implemented: RoPE (``use_nope=False``, K2 heritage), the ungated layer, full-rank q
(``q_lora_rank=None``), padding masks and packed sequences, attention dropout (0 in K3).

Sources (reviewed 2026-09-22): huggingface.co/moonshotai/Kimi-K3 modeling_kimi_linear.py
(KimiRMSNorm :226-236, eager_attention_forward :311-332, KimiMLAAttention :335-474; copy in
../ladders/oss/kimi_k3_hf); vLLM vllm/models/kimi_k3/nvidia/mla.py (norm eps :157/:247/:282,
kv_b_proj split into W_UK/W_UV :436-462, W_UV up-projection :483-489, gate + o_proj :620-637,
W_UK absorption on decode :699-705).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from scratch_llm.k3.config import MLAConfig
from scratch_llm.k3.core.norm import RMSNorm

if TYPE_CHECKING:
    from scratch_llm.k3.model import HybridState

__all__ = ["MLALatentKV", "GatedMLA"]


def _math_dtype(*tensors: Tensor) -> torch.dtype:
    """fp64 if any input is fp64, else fp32: scores and softmax never run in half precision."""
    if any(t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


@dataclass
class MLALatentKV:
    """Latent KV cache for one Gated-MLA layer — the only per-token state MLA keeps.

    ``c_kv`` is the ``kv_a_layernorm``-ed latent; ``k_rope`` is the unrotated key part shared by
    all heads (never normed). Per-head K/V are never materialized: the absorbed decode path folds
    ``kv_b_proj`` into the query and the output. Grows by one row per decoded token; at full scale
    that is (512 + 64) × 2 bytes per token per MLA layer in bf16.
    """

    c_kv: Tensor  # (B, T, kv_lora_rank)
    k_rope: Tensor  # (B, T, qk_rope_head_dim)

    @classmethod
    def empty(
        cls, cfg: MLAConfig, batch: int, device: torch.device, dtype: torch.dtype
    ) -> MLALatentKV:
        return cls(
            torch.zeros(batch, 0, cfg.kv_lora_rank, device=device, dtype=dtype),
            torch.zeros(batch, 0, cfg.qk_rope_head_dim, device=device, dtype=dtype),
        )

    @property
    def length(self) -> int:
        return self.c_kv.shape[1]


class GatedMLA(nn.Module):
    """One K3 Gated-MLA layer: forward(x [B, T, hidden], state=None, layer_id=0) -> [B, T, hidden].

    Parameter names are the checkpoint's (``self_attn.*``). With ``state=None`` (training
    prefill) the layer attends over its own T tokens on the naive path. With a HybridState it
    reads ``state.mla_caches[layer_id - 1]`` (None = empty), appends this call's latent, attends
    over cache + new tokens on the absorbed path, and commits the grown cache only after the
    whole forward succeeded.

    ``track_max_logits``: when True, every *training-mode* forward folds the detached max of the
    scaled pre-softmax logit over batch, queries and causal keys into ``last_max_logits``
    [num_heads] as a running max. ``apply_k3_qk_clip`` consumes it (resets it to None), so the clip
    sees the max over every microbatch since the previous step, as torchtitan's ``qk_clip.py``
    (:46, :158) takes the max over its per-microbatch records. Eval-mode forwards leave a pending
    observation untouched; with tracking off it is None.
    """

    def __init__(self, cfg: MLAConfig, hidden_size: int, *, latent_norm_eps: float = 1e-6) -> None:
        super().__init__()
        if not cfg.use_nope:
            raise NotImplementedError("only K3's NoPE MLA is implemented (no rotary embedding)")
        if not cfg.output_gate:
            raise NotImplementedError("only K3's gated MLA is implemented")
        self.cfg = cfg
        self.scale = 1.0 / math.sqrt(cfg.q_head_dim)  # the shared rope dims count
        h, L, R = cfg.num_heads, cfg.kv_lora_rank, cfg.qk_rope_head_dim

        self.q_a_proj = nn.Linear(hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, latent_norm_eps)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, h * cfg.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, L + R, bias=False)
        self.kv_a_layernorm = RMSNorm(L, latent_norm_eps)
        self.kv_b_proj = nn.Linear(L, h * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.o_proj = nn.Linear(h * cfg.v_head_dim, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, h * cfg.v_head_dim, bias=False)  # full-rank gate

        self.track_max_logits = False
        self.last_max_logits: Tensor | None = None

    def _read_cache(self, state: HybridState, layer_id: int, x: Tensor) -> MLALatentKV:
        if layer_id < 1:
            raise ValueError("stateful forward needs the 1-based layer_id")
        cache = state.mla_caches[layer_id - 1]
        B = x.shape[0]
        if cache is None:
            return MLALatentKV.empty(self.cfg, B, x.device, x.dtype)
        P, L, R = cache.length, self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim
        if cache.c_kv.shape != (B, P, L) or cache.k_rope.shape != (B, P, R):
            raise ValueError("MLALatentKV does not match this layer and batch; reset the cache")
        return cache

    def _project(self, x: Tensor) -> tuple[Tensor, MLALatentKV]:
        """x [B, T, hidden] -> q [B, T, h, nope + rope] and this call's latent: c_kv normed
        [B, T, kv_lora], k_rope raw [B, T, rope]."""
        B, T, _ = x.shape
        cfg = self.cfg
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, T, cfg.num_heads, cfg.q_head_dim)
        latent = self.kv_a_proj_with_mqa(x)
        c_kv, k_rope = latent.split((cfg.kv_lora_rank, cfg.qk_rope_head_dim), dim=-1)
        return q, MLALatentKV(self.kv_a_layernorm(c_kv), k_rope)  # k_rope: no norm, no rotation

    def _causal_softmax(self, logits: Tensor) -> Tensor:
        """Scaled logits [B, h, T, S] of the last T of S positions -> probabilities over
        s <= (S - T) + t. Records ``last_max_logits`` after the mask, before the softmax."""
        T, S = logits.shape[-2:]
        allowed = torch.ones(T, S, dtype=torch.bool, device=logits.device).tril(S - T)
        logits = logits.masked_fill(~allowed, -math.inf)
        if not self.track_max_logits:
            self.last_max_logits = None
        elif self.training:
            s_max = logits.detach().amax(dim=(0, 2, 3))
            prev = self.last_max_logits
            self.last_max_logits = s_max if prev is None else torch.maximum(prev, s_max)
        return logits.softmax(dim=-1)

    def attend_naive(self, q: Tensor, kv: MLALatentKV) -> Tensor:
        """Materialize per-head K/V from the latent, then attend (HF KimiMLAAttention).

        q [B, T, h, nope + rope] are the queries of the last T of the kv.length positions.
        Returns the pre-gate output [B, T, h * v] in the math dtype.
        """
        cfg = self.cfg
        B, T, h, _ = q.shape
        nope, R, V = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        kv_up = self.kv_b_proj(kv.c_kv)  # a module-dtype matmul, like every projection
        with torch.autocast(q.device.type, enabled=False):
            dtype = _math_dtype(q, kv_up)
            q_nope, q_rope = q.to(dtype).split((nope, R), dim=-1)
            k_nope, v = kv_up.to(dtype).view(B, kv.length, h, nope + V).split((nope, V), dim=-1)
            logits = torch.einsum("bthd,bshd->bhts", q_nope, k_nope)
            # One k_rope row per position, read by every head (zero when rope = 0).
            logits = logits + torch.einsum("bthr,bsr->bhts", q_rope, kv.k_rope.to(dtype))
            probs = self._causal_softmax(logits * self.scale)
            o = torch.einsum("bhts,bshv->bthv", probs, v)
        return o.reshape(B, T, h * V)

    def attend_absorbed(self, q: Tensor, kv: MLALatentKV) -> Tensor:
        """Attend in the latent space: W_UK folded into the query, W_UV applied after the
        latent-weighted sum (vLLM decode). Reads only the latent; never builds per-head K/V.

        Same contract as :meth:`attend_naive`.
        """
        cfg = self.cfg
        B, T, h, _ = q.shape
        nope, R, V = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
        with torch.autocast(q.device.type, enabled=False):
            dtype = _math_dtype(q, kv.c_kv, self.kv_b_proj.weight)
            w = self.kv_b_proj.weight.to(dtype).view(h, nope + V, cfg.kv_lora_rank)
            w_uk, w_uv = w.split((nope, V), dim=1)  # [h, nope, kv_lora], [h, v, kv_lora]
            q_nope, q_rope = q.to(dtype).split((nope, R), dim=-1)
            c = kv.c_kv.to(dtype)
            q_latent = torch.einsum("bthd,hdl->bthl", q_nope, w_uk)  # W_UK^T q_nope
            logits = torch.einsum("bthl,bsl->bhts", q_latent, c)
            logits = logits + torch.einsum("bthr,bsr->bhts", q_rope, kv.k_rope.to(dtype))
            probs = self._causal_softmax(logits * self.scale)
            o_latent = torch.einsum("bhts,bsl->bthl", probs, c)  # sum_s p_s c_s
            o = torch.einsum("bthl,hvl->bthv", o_latent, w_uv)  # W_UV (sum_s p_s c_s)
        return o.reshape(B, T, h * V)

    def forward(
        self,
        x: Tensor,  # (B, T, hidden) — already RMSNorm-ed by the block
        state: HybridState | None = None,
        layer_id: int = 0,
    ) -> Tensor:
        past = None if state is None else self._read_cache(state, layer_id, x)
        q, new = self._project(x)
        if past is None:  # training prefill: T queries over the same T keys
            kv = new
            o = self.attend_naive(q, kv)
        else:  # decode or extend: attend over cache + new tokens in the latent space
            kv = MLALatentKV(
                torch.cat((past.c_kv.detach(), new.c_kv), dim=1),
                torch.cat((past.k_rope.detach(), new.k_rope), dim=1),
            )
            o = self.attend_absorbed(q, kv)

        gate = self.g_proj(x).to(o.dtype).sigmoid()  # reads the layer input, not the attention
        out = self.o_proj((o * gate).to(x.dtype))
        if state is not None:  # commit only after the whole forward succeeded
            state.mla_caches[layer_id - 1] = MLALatentKV(kv.c_kv.detach(), kv.k_rope.detach())
        return out
