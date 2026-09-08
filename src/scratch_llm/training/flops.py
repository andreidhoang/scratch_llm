"""T1/T-R0 — the arithmetic under `tok/s` and `MFU`, with every convention named.

An MFU number is three choices wearing one name. Two runs can report 40% and 55% on identical
hardware doing identical work and both be honest, because they differ in:

  1. **Which FLOPs are in the numerator.** The famous `C ≈ 6·N·D` counts *parameters*: 2 FLOPs per
     param forward (one multiply, one add), 4 back (input-grad matmul + weight-grad matmul, each
     the same shape as the forward). It therefore sees only work whose cost scales with N — the
     linear layers. The `QK^T` and `PV` contractions have no parameters at all: their cost scales
     with sequence length, and per *token* it is linear in S (per *sequence*, quadratic). At
     Llama-3-1B shape they are 9% of the total at S=2048 and 30% at S=8192 — a rounding error at
     the first and a third of the answer at the second. `AttentionTerm` below is that term, kept
     explicit so a number quoted here is comparable to one quoted elsewhere only when the caller
     says which convention it used.
  2. **Whether causal sparsity is credited.** A causal kernel (FlashAttention, FlexAttention with
     a causal mask) skips the fully-masked blocks and executes about half the attention FLOPs the
     dense formula counts. torchtitan deliberately does *not* halve
     (`oss/torchtitan/torchtitan/models/utils.py:437`: "we follow the convention and do not account
     for sparsity in causal attention"), so its attention term is ~2x the work its FlexAttention
     kernel actually issues. `causal_factor=1.0` reproduces torchtitan; `0.5` is the executed-work
     estimate. Neither is wrong; quoting one as the other is.
  3. **Whether activation recompute is in the numerator.** Activation checkpointing re-runs part of
     the forward inside the backward, so the *hardware* does more FLOPs than the *model* needs.
     MFU (model FLOPs / peak) excludes the recompute and answers "how much of the machine did
     useful model work"; HFU (hardware FLOPs / peak) includes it and answers "how busy was the
     machine". HFU >= MFU always, and the gap is a number the AC policy determines, not an
     unknown. torchtitan reports MFU only (`components/metrics.py:497` divides
     `num_flops_per_token`, which carries no recompute term).

Everything here is integer/float arithmetic on a shape — no torch, no GPU, no measurement. It is
the half of the floor that can be checked on a laptop; `titan_floor.py` supplies the other half
(the stopwatch) and multiplies the two.

Cited torchtitan facts, all at pin d263ca0a under `oss/torchtitan`:
  - `torchtitan/models/llama3/model.py:103`  `return nparams, 6 * active_nparams + attention_op_flops`
  - `torchtitan/models/utils.py:447`         `6 * num_heads * (qk_head_dim + v_head_dim) * attended_tokens`
  - `torchtitan/models/utils.py:519-523`     embeddings excluded from active params unless tied to lm_head
  - `torchtitan/components/metrics.py:485`   `tps = ntokens / (time * non_data_parallel_size)` — per device
  - `torchtitan/components/metrics.py:497`   `mfu = 100 * num_flops_per_token * tps / gpu_peak_flops`
  - `torchtitan/tools/utils.py:173`          H100 SXM peak = 989e12 (dense bf16, spec sheet)
"""

from __future__ import annotations

from dataclasses import dataclass

# --- the 6 in 6ND, decomposed ------------------------------------------------------------------
# One matmul costs 2 FLOPs per parameter per token (multiply + accumulate). The forward does one,
# the backward does two of the same shape (dL/dx and dL/dW). 2 + 4 = 6.
FWD_FLOPS_PER_PARAM = 2.0
BWD_FLOPS_PER_PARAM = 4.0
FLOPS_PER_PARAM = FWD_FLOPS_PER_PARAM + BWD_FLOPS_PER_PARAM  # 6

# The same 2-fwd/4-bwd split applied to the two parameterless attention contractions
# (QK^T and PV): torchtitan writes it as one constant 6 (models/utils.py:447).
ATTENTION_FLOPS_PER_ELEMENT = 6.0
ATTENTION_FWD_FRACTION = FWD_FLOPS_PER_PARAM / FLOPS_PER_PARAM  # 1/3 of the 6 is forward

# Vendor spec-sheet dense bf16 peaks, FLOP/s, *as torchtitan reads them* — mirrored from
# `oss/torchtitan/torchtitan/tools/utils.py:143-195` so an MFU computed here has the same
# denominator as the one torchtitan prints. (scratch_llm.utils.mfu carries 989.5e12 for H100 SXM
# from NVIDIA's sheet; torchtitan rounds to 989e12. A 0.05% denominator difference is invisible
# next to run-to-run noise, but the floor is a comparison, so use torchtitan's number.)
TORCHTITAN_PEAK_BF16_DENSE: dict[str, float] = {
    "H100 SXM": 989e12,  # tools/utils.py:173  (H100 NVL 835e12, H100 PCIe 756e12)
    "H200": 989e12,  # tools/utils.py:176
    "A100": 312e12,  # tools/utils.py:159
    "B200": 2.25e15,  # tools/utils.py:195
    "GB200": 2.5e15,  # tools/utils.py:191
}

# Activation-checkpointing policies, named exactly as torchtitan's classes in
# `torchtitan/distributed/activation_checkpoint.py`.
AC_NONE = "none"  # ac_config = None: nothing recomputed
AC_FULL = "full"  # FullAC (:167) — ptd_checkpoint_wrapper on the whole block
AC_SELECTIVE_OP = "selective_op"  # SelectiveAC (:186) — save compute ops, recompute every 2nd mm
AC_POLICIES = (AC_NONE, AC_FULL, AC_SELECTIVE_OP)


def _positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


@dataclass(frozen=True)
class LinearSpec:
    """One matmul-bearing weight: its name, its shape, and therefore its parameter count."""

    name: str
    in_features: int
    out_features: int

    @property
    def params(self) -> int:
        return self.in_features * self.out_features


@dataclass(frozen=True)
class DecoderShape:
    """A dense decoder as the FLOP model sees it: counts, not modules.

    Field names follow the torchtitan config they are read from
    (`torchtitan/models/llama3/__init__.py`), so a shape here can be checked against that file by
    eye. ``tie_embeddings`` is load-bearing for the parameter count *and* for the FLOP count: with
    tying, ``tok_embeddings.weight is lm_head.weight`` (`models/common/decoder.py:231`), the shared
    tensor appears once in ``named_parameters()``, and torchtitan's exclusion of embedding tables
    from the active count does not apply to it (`models/utils.py:519-523`) — it is counted, once,
    because the lm_head really does run a matmul against it on every token.
    """

    n_layers: int
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    vocab_size: int
    tie_embeddings: bool
    norms_per_layer: int = 2  # attention_norm + ffn_norm, RMSNorm: `dim` weights each
    final_norm: bool = True

    # --- parameters ----------------------------------------------------------------------------
    def block_linears(self) -> tuple[LinearSpec, ...]:
        """The block's matmul weights **in forward execution order**.

        Order matters and is not cosmetic: torchtitan's selective AC recomputes *every second* mm
        by position within the block's forward (`activation_checkpoint.py:272-278`), so which
        weight is recomputed is decided by this sequence. It is
        wqkv -> wo -> w1 -> w3 -> w2, from `models/common/attention.py:765` (fused qkv, one mm),
        the attention output projection, and `models/common/feed_forward.py:54`
        (``w2(silu(w1(x)) * w3(x))`` — Python evaluates ``w1`` before ``w3``).
        """
        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv_heads * self.head_dim
        return (
            LinearSpec("wqkv", self.dim, q_out + 2 * kv_out),
            LinearSpec("wo", q_out, self.dim),
            LinearSpec("w1", self.dim, self.ffn_hidden),
            LinearSpec("w3", self.dim, self.ffn_hidden),
            LinearSpec("w2", self.ffn_hidden, self.dim),
        )

    @property
    def block_linear_params(self) -> int:
        return sum(spec.params for spec in self.block_linears())

    @property
    def block_norm_params(self) -> int:
        return self.norms_per_layer * self.dim

    @property
    def block_params(self) -> int:
        return self.block_linear_params + self.block_norm_params

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.dim

    @property
    def lm_head_params(self) -> int:
        return self.vocab_size * self.dim

    @property
    def total_params(self) -> int:
        """Every parameter, counting a tied embedding/lm_head tensor once (as PyTorch does)."""
        shared = self.embedding_params if self.tie_embeddings else 0
        return (
            self.n_layers * self.block_params
            + self.embedding_params
            + self.lm_head_params
            - shared
            + (self.dim if self.final_norm else 0)
        )

    @property
    def active_params(self) -> int:
        """torchtitan's ``active_nparams`` — the N in 6ND (`models/utils.py:531-535`).

        Untied embedding tables are dropped (a lookup is not a matmul); a tied table is kept, once.
        Norm weights are *not* dropped: the convention weights every remaining parameter by 1, so
        67,584 RMSNorm weights ride along in a 1.24B count. That is 0.005% and is left alone here
        so the number matches torchtitan's exactly — see ``matmul_params`` for the strict reading.
        """
        embedding = self.embedding_params if self.tie_embeddings else 0
        return (
            self.n_layers * self.block_params
            + embedding
            + (0 if self.tie_embeddings else self.lm_head_params)
            + (self.dim if self.final_norm else 0)
        )

    @property
    def matmul_params(self) -> int:
        """Only weights consumed by a matmul: the block linears plus the (possibly tied) lm_head.

        ``active_params - matmul_params`` is exactly the norm-weight count.
        """
        return self.n_layers * self.block_linear_params + self.lm_head_params

    # --- FLOPs ---------------------------------------------------------------------------------
    def dense_flops_per_token(self, *, params: int | None = None) -> float:
        """The 6ND term: ``6 * active_params``, per token. No sequence-length dependence."""
        n = self.active_params if params is None else params
        return FLOPS_PER_PARAM * n

    def attention_flops_per_token(self, seq_len: int, *, causal_factor: float = 1.0) -> float:
        """The parameterless term: ``6 * H * (d_qk + d_v) * S`` per layer, per token.

        Reproduces `torchtitan/models/utils.py:447` at ``causal_factor=1.0``. Per token it is
        *linear* in S; multiply by the S tokens of a sequence and the familiar S^2 appears. Uses
        ``n_heads`` (query heads), not ``n_kv_heads``: GQA shrinks the KV *cache*, not the number
        of query-head score matrices computed.

        ``causal_factor`` is the one knob torchtitan does not have. 1.0 counts the full S x S
        score matrix (torchtitan's convention); 0.5 is the causal kernel's executed work.
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if not 0.0 < causal_factor <= 1.0:
            raise ValueError(f"causal_factor must be in (0, 1], got {causal_factor}")
        qk_v_dims = 2 * self.head_dim  # qk_head_dim + v_head_dim
        per_layer = ATTENTION_FLOPS_PER_ELEMENT * self.n_heads * qk_v_dims * seq_len
        return self.n_layers * per_layer * causal_factor


def recomputed_linears(shape: DecoderShape, policy: str) -> tuple[LinearSpec, ...]:
    """Which block linears the AC policy re-runs in the backward, per `activation_checkpoint.py`.

    - ``AC_NONE``: none.
    - ``AC_FULL`` (:167): the whole block is re-run, so every linear in it.
    - ``AC_SELECTIVE_OP`` (:186): the policy at :265-279 counts mm/linear ops as they execute and
      returns ``PREFER_RECOMPUTE`` when the running count is even, ``MUST_SAVE`` otherwise. The
      counter is incremented before the parity test (:272 then :276), so op 1 is saved, op 2 is
      recomputed, op 3 saved ... i.e. the even positions of ``block_linears()``. For a fused-qkv
      Llama-3 block that is {wo, w3}: 2 of 5 mms, and 34% of the block's linear FLOPs, not 50% —
      the mms are not the same size, and the parity does not care.

      Caveat, stated because it bounds what this function can promise: ``PREFER_RECOMPUTE`` is
      advisory. Under ``torch.compile`` the AOTAutograd min-cut partitioner may choose to save a
      preferred-recompute value anyway; only ``MUST_SAVE`` is binding. So this is the policy's
      *nominal* recompute set — an upper bound on executed recompute FLOPs, exact in eager.
    """
    if policy not in AC_POLICIES:
        raise ValueError(f"unknown AC policy {policy!r}; expected one of {AC_POLICIES}")
    linears = shape.block_linears()
    if policy == AC_NONE:
        return ()
    if policy == AC_FULL:
        return linears
    return tuple(spec for i, spec in enumerate(linears, start=1) if i % 2 == 0)


@dataclass(frozen=True)
class FlopBreakdown:
    """Per-token FLOPs of one training step, split by which convention includes which term."""

    dense_per_token: float  # 6 * active_params        — the 6ND term (linears; no S dependence)
    attention_per_token: float  # 6 * H * 2d * S * causal  — the term 6ND omits
    recompute_per_token: float  # the AC tax: extra forwards the hardware runs, the model does not
    seq_len: int
    causal_factor: float
    ac_policy: str

    @property
    def model_per_token(self) -> float:
        """MFU numerator: what the math requires. torchtitan's ``num_flops_per_token``."""
        return self.dense_per_token + self.attention_per_token

    @property
    def executed_per_token(self) -> float:
        """HFU numerator: what the hardware runs, recompute included."""
        return self.model_per_token + self.recompute_per_token

    @property
    def ac_multiplier(self) -> float:
        """``executed / model`` — 1.0 with no AC, and the HFU/MFU ratio for this config."""
        return self.executed_per_token / self.model_per_token

    @property
    def attention_share(self) -> float:
        """Fraction of model FLOPs the 6ND approximation drops. Grows linearly in S."""
        return self.attention_per_token / self.model_per_token


def flop_breakdown(
    shape: DecoderShape,
    seq_len: int,
    *,
    ac_policy: str = AC_NONE,
    causal_factor: float = 1.0,
) -> FlopBreakdown:
    """Per-token FLOPs for ``shape`` at ``seq_len`` under ``ac_policy``.

    The recompute term is a *forward* only (``2`` FLOPs/param, and 1/3 of the attention 6): AC
    re-runs the forward to rebuild activations, it does not re-run the backward. Under ``AC_FULL``
    the re-run block includes its attention contractions; under ``AC_SELECTIVE_OP`` it does not —
    SDPA and FlexAttention outputs are in the MUST_SAVE set
    (`activation_checkpoint.py:42-49`), so attention is never recomputed there. Embedding, final
    norm and lm_head sit outside the checkpointed blocks (`activation_checkpoint.py:156-161` wraps
    ``model.layers`` only) and are never recomputed under either policy.
    """
    if ac_policy not in AC_POLICIES:
        raise ValueError(f"unknown AC policy {ac_policy!r}; expected one of {AC_POLICIES}")
    dense = shape.dense_flops_per_token()
    attention = shape.attention_flops_per_token(seq_len, causal_factor=causal_factor)

    recompute_params = sum(spec.params for spec in recomputed_linears(shape, ac_policy))
    recompute = FWD_FLOPS_PER_PARAM * shape.n_layers * recompute_params
    if ac_policy == AC_FULL:
        # The whole block re-runs: its norms and its attention contractions come with it.
        recompute += FWD_FLOPS_PER_PARAM * shape.n_layers * shape.block_norm_params
        recompute += ATTENTION_FWD_FRACTION * attention
    return FlopBreakdown(
        dense_per_token=dense,
        attention_per_token=attention,
        recompute_per_token=recompute,
        seq_len=seq_len,
        causal_factor=causal_factor,
        ac_policy=ac_policy,
    )


def utilization(flops_per_token: float, tokens_per_s: float, peak_flops: float) -> float:
    """``flops_per_token * tokens_per_s / peak_flops`` — dimensionless, 1.0 at peak.

    Per *device*: feed it torchtitan's per-device ``tps`` (`components/metrics.py:485` divides the
    rank-local token count by ``cp*tp*pp``, so TP peers do not double-count) against one GPU's
    peak. Feeding a cluster's tok/s against one GPU's peak is the most common way to publish a
    wrong MFU.
    """
    _positive("tokens_per_s", tokens_per_s)
    _positive("peak_flops", peak_flops)
    if flops_per_token < 0:
        raise ValueError(f"flops_per_token must be >= 0, got {flops_per_token}")
    return flops_per_token * tokens_per_s / peak_flops


@dataclass(frozen=True)
class StepReport:
    """One throughput measurement, reported under every convention at once.

    Four numbers from one stopwatch reading. They differ by factors this config determines, so a
    disagreement with someone else's MFU is a disagreement about conventions, not about hardware:

      ``mfu_6nd``       6ND only — what you get if you forget attention exists.
      ``mfu_model``     6ND + attention, causal_factor as configured. **torchtitan's mfu(%)/100.**
      ``hfu``           model + AC recompute: how busy the chip was.
      ``mfu_executed``  6ND + attention at causal_factor 0.5 — closest to work actually issued.
    """

    tokens_per_s_per_device: float
    peak_flops_per_device: float
    breakdown: FlopBreakdown
    shape: DecoderShape

    @property
    def mfu_6nd(self) -> float:
        return utilization(
            self.breakdown.dense_per_token,
            self.tokens_per_s_per_device,
            self.peak_flops_per_device,
        )

    @property
    def mfu_model(self) -> float:
        return utilization(
            self.breakdown.model_per_token,
            self.tokens_per_s_per_device,
            self.peak_flops_per_device,
        )

    @property
    def hfu(self) -> float:
        return utilization(
            self.breakdown.executed_per_token,
            self.tokens_per_s_per_device,
            self.peak_flops_per_device,
        )

    @property
    def mfu_executed_attention(self) -> float:
        causal = flop_breakdown(
            self.shape,
            self.breakdown.seq_len,
            ac_policy=self.breakdown.ac_policy,
            causal_factor=0.5,
        )
        return utilization(
            causal.model_per_token,
            self.tokens_per_s_per_device,
            self.peak_flops_per_device,
        )

    @property
    def achieved_tflops_per_device(self) -> float:
        """torchtitan's ``tflops`` column (`components/metrics.py:493`): model FLOPs, not executed."""
        return self.breakdown.model_per_token * self.tokens_per_s_per_device / 1e12

    def as_dict(self) -> dict[str, float | str]:
        return {
            "tps_per_device": self.tokens_per_s_per_device,
            "peak_flops_per_device": self.peak_flops_per_device,
            "seq_len": self.breakdown.seq_len,
            "ac_policy": self.breakdown.ac_policy,
            "causal_factor": self.breakdown.causal_factor,
            "active_params": self.shape.active_params,
            "total_params": self.shape.total_params,
            "flops_per_token_6nd": self.breakdown.dense_per_token,
            "flops_per_token_attention": self.breakdown.attention_per_token,
            "flops_per_token_model": self.breakdown.model_per_token,
            "flops_per_token_executed": self.breakdown.executed_per_token,
            "attention_share_of_model": self.breakdown.attention_share,
            "ac_multiplier": self.breakdown.ac_multiplier,
            "tflops_per_device": self.achieved_tflops_per_device,
            "mfu_6nd_only": self.mfu_6nd,
            "mfu_model_torchtitan": self.mfu_model,
            "mfu_model_causal_half": self.mfu_executed_attention,
            "hfu_executed": self.hfu,
        }


def step_report(
    shape: DecoderShape,
    *,
    seq_len: int,
    tokens_per_s_per_device: float,
    peak_flops_per_device: float,
    ac_policy: str = AC_NONE,
    causal_factor: float = 1.0,
) -> StepReport:
    """Assemble a `StepReport` from a shape, a stopwatch reading, and a peak."""
    return StepReport(
        tokens_per_s_per_device=tokens_per_s_per_device,
        peak_flops_per_device=peak_flops_per_device,
        breakdown=flop_breakdown(shape, seq_len, ac_policy=ac_policy, causal_factor=causal_factor),
        shape=shape,
    )


# --- the rung's pinned model -------------------------------------------------------------------
# Llama-3 1B exactly as `oss/torchtitan/torchtitan/models/llama3/__init__.py:151-195` builds it
# (flavor "1B"). ffn_hidden is `compute_ffn_hidden_dim(2048, multiple_of=1024,
# ffn_dim_multiplier=1.5)` = int(1.5 * int(2*4*2048/3)) rounded up to 1024 = 8192
# (`models/common/feed_forward.py:28-31`). Weight tying is on, so the 128256 x 2048 table is one
# tensor and is counted once, in both the parameter total and the 6N term.
LLAMA3_1B = DecoderShape(
    n_layers=16,
    dim=2048,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,  # dim // n_heads
    ffn_hidden=8192,
    vocab_size=128256,
    tie_embeddings=True,
)
