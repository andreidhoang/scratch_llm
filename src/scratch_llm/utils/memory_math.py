"""Training-memory algebra — the closed-form accounting behind "how would you train a 100B model?".

A2 systems (`optimizer_state_sharding_accounting`). Pure integer arithmetic, no torch: every number
in `docs/design/A2_100B_MEMORY_ONEPAGER.md` is produced by a function here and pinned by
`tests/test_memory_math.py` — the doc is generated output, not hand-typed folklore.

The two stories these functions encode:

- **16 B/param (full fp32):** weight 4 + grad 4 + Adam m 4 + Adam v 4. At 100B params that is
  1.6 TB of state before a single activation — ~20 H100-80GBs of pure state.
- **18–20 B/param (bf16 mixed precision):** fp32 master 4 + fp32 grad 4 + Adam m 4 + Adam v 4
  (the optimizer-side state, 16) plus the bf16 compute copies: weight 2, and grad 2 when backward
  keeps a separate bf16 grad buffer (20 B) vs fusing straight into the fp32 grad (18 B). Mixed
  precision makes the *matmuls* cheaper, not the state — the state gets **bigger**.

ZeRO shards that state along the natural bucket boundaries: stage 1 shards the optimizer bucket
(master+m+v+fp32-grad), stage 2 also shards the backward grads, stage 3 also shards the weights.

Falsifiable invariants (tested): `zero_shard_bytes(stage, N, 1, mixed) == optimizer_state_bytes(N,
mixed)` for every stage; per-rank bytes are monotone non-increasing in stage; stage 3 divides the
total exactly by W (when W | N); the activation formula reproduces Korthikanti et al. (2022)
`sbh·(34 + 5·a·s/h)` and flash removes exactly the `5·a·s²·b` term.

Interview: "Why can't one GPU train a 100B model, and what does each ZeRO stage buy you?" —
`optimizer_state_bytes` / `zero_shard_bytes` are that answer as executable arithmetic.
"""

from __future__ import annotations

import math
from typing import Literal

BYTES_FP32 = 4
BYTES_BF16 = 2
GB = 10**9  # decimal, vendor-style: an "80 GB" H100 budget is 80 * GB bytes
TB = 10**12

ZeroStage = Literal[0, 1, 2, 3]  # 0 = plain DDP baseline (nothing sharded)


def param_bytes(n_params: int, dtype_bytes: int = BYTES_FP32) -> int:
    """Bytes to hold one copy of the weights: ``n_params * dtype_bytes``."""
    if n_params < 0 or dtype_bytes <= 0:
        raise ValueError("n_params must be >= 0 and dtype_bytes > 0")
    return n_params * dtype_bytes


def training_state_breakdown(
    n_params: int, mixed_precision: bool, *, bf16_grad_copy: bool = True
) -> dict[str, int]:
    """Per-term training-state bytes (weights + grads + Adam state), keyed by term.

    fp32 story (``mixed_precision=False``, 16 B/param):
      ``fp32_weights`` 4 + ``fp32_grads`` 4 + ``adam_m_fp32`` 4 + ``adam_v_fp32`` 4.

    Mixed story (``mixed_precision=True``, 18–20 B/param):
      ``fp32_master_weights`` 4 + ``fp32_grads`` 4 + ``adam_m_fp32`` 4 + ``adam_v_fp32`` 4
      + ``bf16_weights`` 2 (the compute copy the fwd/bwd matmuls read)
      + ``bf16_grads`` 2 iff ``bf16_grad_copy`` (backward's output buffer before the fp32
      upcast; some stacks fuse the upcast and never materialize it → the 18 B variant).

    The fp32 grad sits with the optimizer terms because it exists to feed the fp32 Adam step —
    under ZeRO it is materialized only for the rank's owned shard.
    """
    if n_params < 0:
        raise ValueError("n_params must be >= 0")
    if not mixed_precision:
        return {
            "fp32_weights": BYTES_FP32 * n_params,
            "fp32_grads": BYTES_FP32 * n_params,
            "adam_m_fp32": BYTES_FP32 * n_params,
            "adam_v_fp32": BYTES_FP32 * n_params,
        }
    terms = {
        "fp32_master_weights": BYTES_FP32 * n_params,
        "fp32_grads": BYTES_FP32 * n_params,
        "adam_m_fp32": BYTES_FP32 * n_params,
        "adam_v_fp32": BYTES_FP32 * n_params,
        "bf16_weights": BYTES_BF16 * n_params,
    }
    if bf16_grad_copy:
        terms["bf16_grads"] = BYTES_BF16 * n_params
    return terms


def optimizer_state_bytes(
    n_params: int, mixed_precision: bool, *, bf16_grad_copy: bool = True
) -> int:
    """Total training-state bytes: 16 B/param fp32, 18–20 B/param mixed (see breakdown above)."""
    return sum(
        training_state_breakdown(n_params, mixed_precision, bf16_grad_copy=bf16_grad_copy).values()
    )


def activation_bytes_per_layer(
    batch: int, seq: int, d_model: int, n_heads: int, flash: bool
) -> int:
    """Saved-for-backward activation bytes for ONE transformer block: ``s·b·h·34 + 5·a·s²·b``.

    The residual-stream accounting of Korthikanti et al. 2022 (§4.1), the standard reference:
    2-byte (bf16/fp16) activations, 1-byte dropout masks, GPT block with a 4·d_model MLP, no
    tensor/sequence parallelism. The linear term ``34·s·b·h`` bytes is everything proportional to
    the residual stream (LN inputs, Q/K/V, attn out, the 8h of MLP up+GeLU, masks); the quadratic
    term ``5·a·s²·b`` bytes is the two materialized (seq × seq) attention matrices (scores fp16 2 +
    softmax fp16 2 + mask 1). ``flash=True`` drops exactly that quadratic term — FlashAttention
    never materializes the score/prob matrices and recomputes them tile-wise in backward.
    """
    if min(batch, seq, d_model, n_heads) <= 0:
        raise ValueError("batch, seq, d_model, n_heads must all be > 0")
    linear = 34 * seq * batch * d_model
    quadratic = 0 if flash else 5 * n_heads * seq * seq * batch
    return linear + quadratic


def residual_stream_bytes(batch: int, seq: int, d_model: int, dtype_bytes: int = BYTES_BF16) -> int:
    """Bytes of ONE block-boundary residual tensor ``(batch, seq, d_model)`` — what gradient
    checkpointing keeps per layer instead of the full ~34·s·b·h interior."""
    if min(batch, seq, d_model) <= 0 or dtype_bytes <= 0:
        raise ValueError("batch, seq, d_model, dtype_bytes must all be > 0")
    return batch * seq * d_model * dtype_bytes


def _zero_buckets(mixed_precision: bool, bf16_grad_copy: bool) -> tuple[int, int, int]:
    """(weights, grads, optimizer) bytes/param — the three ZeRO shard boundaries."""
    if mixed_precision:
        weights = BYTES_BF16  # bf16 compute copy is what fwd/bwd touch
        grads = BYTES_BF16 if bf16_grad_copy else 0
        optimizer = 4 * BYTES_FP32  # master + fp32 grad + m + v
    else:
        weights = BYTES_FP32
        grads = BYTES_FP32
        optimizer = 2 * BYTES_FP32  # m + v (the fp32 weight IS the master)
    return weights, grads, optimizer


def zero_shard_breakdown(
    stage: ZeroStage,
    n_params: int,
    world_size: int,
    mixed_precision: bool,
    *,
    bf16_grad_copy: bool = True,
) -> dict[str, int]:
    """Per-RANK bytes under ZeRO ``stage``, split into the three buckets.

    - stage 0: plain DDP — weights, grads, optimizer state all replicated on every rank.
    - stage 1: optimizer bucket sharded (each rank owns ~N/W params' master+fp32-grad+m+v);
      weights + backward grads stay replicated (grads must, for the all-reduce).
    - stage 2: grads also sharded (reduce-scatter instead of all-reduce).
    - stage 3 (FSDP): weights also sharded — everything is ~1/W, at the price of all-gathering
      weights on demand every fwd/bwd.

    Sharded buckets charge ``ceil(N/W)`` owned params (the greedy-partition granularity), so
    ``W | N`` gives exact division.
    """
    if stage not in (0, 1, 2, 3):
        raise ValueError(f"stage must be 0, 1, 2 or 3, got {stage}")
    if n_params < 0 or world_size < 1:
        raise ValueError("n_params must be >= 0 and world_size >= 1")
    weights, grads, optimizer = _zero_buckets(mixed_precision, bf16_grad_copy)
    owned = math.ceil(n_params / world_size) if n_params else 0
    return {
        "weights": weights * (owned if stage >= 3 else n_params),
        "grads": grads * (owned if stage >= 2 else n_params),
        "optimizer": optimizer * (owned if stage >= 1 else n_params),
    }


def zero_shard_bytes(
    stage: ZeroStage,
    n_params: int,
    world_size: int,
    mixed_precision: bool,
    *,
    bf16_grad_copy: bool = True,
) -> int:
    """Total per-rank training-state bytes under ZeRO ``stage`` (sum of the bucket breakdown)."""
    return sum(
        zero_shard_breakdown(
            stage, n_params, world_size, mixed_precision, bf16_grad_copy=bf16_grad_copy
        ).values()
    )


def gpus_needed(
    n_params: int,
    hbm_bytes: int,
    *,
    stage: ZeroStage = 3,
    mixed_precision: bool = True,
    bf16_grad_copy: bool = True,
    activation_bytes_per_gpu: int = 0,
) -> int:
    """Smallest world size whose per-rank ZeRO-``stage`` state (+ an optional per-GPU activation
    budget) fits in ``hbm_bytes``. Raises ValueError when NO world size fits — i.e. the replicated
    buckets alone exceed HBM, the "ZeRO-1 cannot rescue 100B on 80 GB" fact."""
    if hbm_bytes <= 0:
        raise ValueError("hbm_bytes must be > 0")
    budget = hbm_bytes - activation_bytes_per_gpu
    weights, grads, optimizer = _zero_buckets(mixed_precision, bf16_grad_copy)
    per_param_sharded = (
        (weights if stage >= 3 else 0)
        + (grads if stage >= 2 else 0)
        + (optimizer if stage >= 1 else 0)
    )
    replicated = (weights + grads + optimizer - per_param_sharded) * n_params
    shard_budget = budget - replicated
    if per_param_sharded == 0:
        if replicated > budget:
            raise ValueError("replicated state exceeds HBM at every world size")
        return 1
    max_owned = shard_budget // per_param_sharded
    if max_owned < 1:
        raise ValueError("replicated state exceeds HBM at every world size")
    return max(math.ceil(n_params / max_owned), 1)
