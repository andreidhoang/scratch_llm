"""The one config both sides of T1/T-R1 are measured at — ours and T-R0's torchtitan floor.

Invariant 2 of the workspace says a number is only a number against a floor at *matched* shape
and dtype. For a kernel that means (M, N, K) and bf16. For a training step it means far more:
model shape, sequence length, tokens per optimizer step, parameter and reduction dtypes, the
parallel degrees, the activation-checkpointing policy, whether the model is compiled, and how
many steps are discarded before the stopwatch starts. Each of those moves tok/s by tens of
percent, and any one of them silently differing between our run and torchtitan's makes "85% of
torchtitan" a comparison to nothing.

**The floor is the TP=2 member, not the FSDP-only one.** ``experiments/T1/T-R0/spec.md`` measures
two: ``titan_llama3_1b_fsdp8_tps`` (FSDP2 over 8, TP=1 — the plan's §05 T-R0 row) and
``titan_llama3_1b_fsdp4_tp2_tps`` (FSDP2×4 with TP=2 — what the plan's T1 exit line
"≥ 85% torchtitan tok/s (FSDP2+TP=2)" is written against). T-R0's spec hands that choice to this
rung; this module makes it, once, in :data:`T_R1`, and ``floor.sh`` reads it.

Nothing here is re-derived. The shape, the sequence length, the batch, the step budget and the
seed are *imported* from :mod:`scratch_llm.training.titan_floor`, which is the module torchtitan
itself is launched with (``--module scratch_llm.training.titan_floor``). Drift between the two
sides is therefore impossible by construction rather than by vigilance — and
``tests/training/test_t1_t_r1.py`` asserts it anyway, because "impossible by construction" has a
way of surviving exactly one refactor.

FLOP accounting is :mod:`scratch_llm.training.flops`, T-R0's, which is torchtitan's:
``6·active_params + Σ_layers 6·n_heads·2·head_dim·seq_len`` per token. At seq 8192 the attention
term is 30.3% of model FLOPs, so reporting plain 6ND against their 6N+attention would hand us a
free 30%. The AC multiplier under selective AC is 1.063.

Two places our model cannot be byte-identical to theirs. Both are recorded in
``experiments/T1/T-R1/map.md``; both are FLOP-neutral, so the tok/s comparison stands:

1. torchtitan's 1B uses ``scaling="llama"`` RoPE (Llama-3.1 frequency rescaling);
   :class:`~scratch_llm.model.RotaryPositionalEmbedding` is plain RoPE. Same shapes, same FLOPs,
   different values — it moves the loss curve, not the throughput.
2. torchtitan fuses Q/K/V into one ``wqkv`` linear (``fuse_qkv=True``,
   ``models/llama3/__init__.py:178``); ours keeps three projections (``model.py:250-253``). Same
   FLOPs, three launches instead of one and three all-gathers instead of one. This one *can* move
   throughput, and it is the first named candidate if the gap is launch- or comm-bound.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from scratch_llm.model import ModelConfig
from scratch_llm.training.flops import AC_SELECTIVE_OP, LLAMA3_1B, DecoderShape
from scratch_llm.training.titan_floor import (
    LOCAL_BATCH,
    NPROC_PER_NODE,
    SEED,
    SEQ_LEN,
    STEPS,
    TOKENS_PER_MICROBATCH_PER_DP_RANK,
    WARMUP_STEPS,
)

# torchtitan reads the peak off the device at runtime and falls back to this; ours is the same
# number in scratch_llm.utils.mfu.PEAK_FLOPS_BF16_DENSE["H100_SXM"]. Dense bf16, not 2:4-sparse.
H100_SXM_PEAK_BF16 = 989.5e12


@dataclass(frozen=True, kw_only=True)
class T1RungConfig:
    """Everything that must match between our trainer and the torchtitan floor.

    Frozen: a run that mutates its own config produces a number nobody can reproduce. Use
    :func:`dataclasses.replace` for the CPU-sized twin the gloo tests run.
    """

    name: str
    shape: DecoderShape
    seq_len: int

    # --- parallel degrees. THESE ARE NOT THE PARALLELISM DECISION. They are the shape of the
    # world: 8 GPUs and TP=2, from the plan's §05 T-R1 row. *Which* modules get fully_shard at
    # what granularity, how TP composes with the shard axis, and what selective AC saves is the
    # hole in parallel_plan.py.
    world_size: int
    tp_degree: int

    # --- step shape ---
    tokens_per_microbatch_per_dp_rank: int

    # --- precision ---
    param_dtype: str  # "bfloat16" — the all-gathered compute copy
    reduce_dtype: str  # "float32"  — the gradient reduction (mp_policy.py says why)
    ac_policy: str  # flops.AC_* — must match what activation_ckpt.py applies

    # --- measurement hygiene (workspace invariant 4) ---
    total_steps: int
    warmup_steps: int
    seed: int

    # --- the floor this rung's number is a percentage of ---
    floor_preset: str  # a key of titan_floor.PRESETS
    floor_metric: str  # the ledger metric name that preset writes

    def __post_init__(self) -> None:
        if self.world_size % self.tp_degree:
            raise ValueError(
                f"world_size {self.world_size} not divisible by tp_degree {self.tp_degree}"
            )
        if self.tokens_per_microbatch_per_dp_rank % self.seq_len:
            raise ValueError(
                f"tokens_per_microbatch_per_dp_rank {self.tokens_per_microbatch_per_dp_rank} "
                f"is not a whole number of {self.seq_len}-token sequences"
            )
        if self.warmup_steps < 20 or self.measure_steps < 50:
            raise ValueError(
                "workspace invariant 4: >= 20 warm-ups and >= 50 measured iterations; got "
                f"{self.warmup_steps} warm-up and {self.measure_steps} measured"
            )

    # ---- derived shape -------------------------------------------------------------------

    @property
    def dp_degree(self) -> int:
        """FSDP shard degree. The mesh is ``(dp, tp)`` — TP is the inner, fastest-varying axis, so
        a TP group is a pair of adjacent ranks and lands inside one NVLink domain. Reversing the
        two on a multi-node job puts a TP all-reduce on the network; that is the mesh-order bug
        the map calls silent."""
        return self.world_size // self.tp_degree

    @property
    def micro_batch(self) -> int:
        """Sequences per data-parallel rank per forward."""
        return self.tokens_per_microbatch_per_dp_rank // self.seq_len

    @property
    def global_tokens_per_step(self) -> int:
        """Tokens the optimizer sees per step, across the whole world."""
        return self.tokens_per_microbatch_per_dp_rank * self.dp_degree

    @property
    def measure_steps(self) -> int:
        return self.total_steps - self.warmup_steps

    def model_config(self) -> ModelConfig:
        """The ModelConfig our trainer builds.

        ``use_sdpa=True`` so attention is the fused kernel on both sides of the comparison
        (torchtitan's llama3 default backend is Flex/cuDNN SDPA); eager ``(B,H,S,S)`` scores at
        seq 8192 would allocate 8 GiB per layer per microbatch and would not be a matched floor.
        """
        return ModelConfig(
            vocab_size=self.shape.vocab_size,
            d_model=self.shape.dim,
            n_layers=self.shape.n_layers,
            n_heads=self.shape.n_heads,
            n_kv_heads=self.shape.n_kv_heads,
            d_ff=self.shape.ffn_hidden,
            context_length=self.seq_len,
            rope_theta=500000.0,  # torchtitan _1b: ComplexRoPE theta=500000 (__init__.py:172)
            tie_embeddings=self.shape.tie_embeddings,
            use_sdpa=True,
        )

    def as_log_dict(self) -> dict[str, object]:
        """The ``config`` field of the logging spec. Every value that changes the number."""
        return {
            "name": self.name,
            "seq_len": self.seq_len,
            "n_layers": self.shape.n_layers,
            "dim": self.shape.dim,
            "n_heads": self.shape.n_heads,
            "n_kv_heads": self.shape.n_kv_heads,
            "head_dim": self.shape.head_dim,
            "ffn_hidden": self.shape.ffn_hidden,
            "vocab_size": self.shape.vocab_size,
            "tie_embeddings": self.shape.tie_embeddings,
            "world_size": self.world_size,
            "tp_degree": self.tp_degree,
            "dp_degree": self.dp_degree,
            "micro_batch": self.micro_batch,
            "tokens_per_microbatch_per_dp_rank": self.tokens_per_microbatch_per_dp_rank,
            "global_tokens_per_step": self.global_tokens_per_step,
            "param_dtype": self.param_dtype,
            "reduce_dtype": self.reduce_dtype,
            "ac_policy": self.ac_policy,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "seed": self.seed,
            "floor_preset": self.floor_preset,
            "floor_metric": self.floor_metric,
        }


# The rung's config. A number measured at anything else is not this rung's number. Every field
# that also exists on the floor side is imported from titan_floor, never retyped.
T_R1 = T1RungConfig(
    name="T1/T-R1 · llama3-1B shape · 8xH100 · FSDP2(dp=4) x TP=2 · bf16/fp32 · selective AC · compile",
    shape=LLAMA3_1B,
    seq_len=SEQ_LEN,  # 8192
    world_size=NPROC_PER_NODE,  # 8
    tp_degree=2,
    tokens_per_microbatch_per_dp_rank=TOKENS_PER_MICROBATCH_PER_DP_RANK,  # 16384 = 2 x 8192
    param_dtype="bfloat16",
    reduce_dtype="float32",
    ac_policy=AC_SELECTIVE_OP,
    total_steps=STEPS,  # 120
    warmup_steps=WARMUP_STEPS,  # 20 discarded => 100 measured
    seed=SEED,
    floor_preset="fsdp4_tp2",
    floor_metric="titan_llama3_1b_fsdp4_tp2_tps",
)

assert T_R1.micro_batch == LOCAL_BATCH, "local batch drifted from titan_floor"


# A CPU-sized twin for the gloo tests: same *structure* (GQA 4:2, tied embeddings, SwiGLU, TP=2
# divides every sharded axis), one ten-thousandth the size. Every structural property the
# plumbing depends on holds here too, which is what makes a 4-process laptop run a real test of
# the same code path rather than a smoke test of a different one.
T_R1_CPU_SHAPE = DecoderShape(
    n_layers=2,
    dim=32,
    n_heads=4,
    n_kv_heads=2,
    head_dim=8,
    ffn_hidden=64,
    vocab_size=64,
    tie_embeddings=True,
)

T_R1_CPU = replace(
    T_R1,
    name="T1/T-R1 cpu-twin",
    shape=T_R1_CPU_SHAPE,
    seq_len=8,
    world_size=4,
    tp_degree=2,
    tokens_per_microbatch_per_dp_rank=16,
    total_steps=70,
    warmup_steps=20,
)


__all__ = [
    "H100_SXM_PEAK_BF16",
    "T_R1",
    "T_R1_CPU",
    "T_R1_CPU_SHAPE",
    "T1RungConfig",
]
