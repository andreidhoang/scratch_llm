"""The bf16 mixed-precision policy — three dtypes, and why they are not the same one.

FSDP2's :class:`~torch.distributed.fsdp.MixedPrecisionPolicy` names three independently:

- ``param_dtype`` — the dtype the sharded parameters are **all-gathered and computed in**.
  bf16. This is the throughput knob and the comms knob at once: the all-gather moves half the
  bytes of fp32, and the matmuls land on the tensor cores. The fp32 *master* shard stays in the
  optimizer, untouched; only the transient gathered copy is bf16.
- ``reduce_dtype`` — the dtype the gradient **reduce-scatter accumulates in**. fp32, not bf16.
  This is the one people get wrong. A bf16 reduce-scatter over 8 ranks adds 8 numbers with an
  8-bit significand; the relative error of that sum is ~2⁻⁸ ≈ 4e-3 per element, and it is
  *biased* (round-to-nearest-even on a sum of same-signed magnitudes still drifts) rather than
  noise that averages out over a step. The failure mode is not a NaN. It is a loss curve that
  tracks the fp32 run for a thousand steps and then sits 0.02 bpb above it forever — the exact
  shape of bug that costs a week because nothing crashed. Paying fp32 here doubles the
  reduce-scatter bytes and is still cheaper than the all-gather; torchtitan's default is the
  same choice (``torchtitan/distributed/fsdp.py:223-227``, driven by
  ``training.mixed_precision_reduce`` which defaults to float32).
- ``output_dtype`` — left ``None``, so module outputs keep ``param_dtype``. Setting it forces a
  cast at every FSDP unit boundary, which is a real cost and buys nothing here: the loss is
  computed in fp32 anyway because :func:`~scratch_llm.model.cross_entropy` upcasts, and
  :class:`~scratch_llm.model.RMSNorm` already computes in fp32 internally
  (``model.py:158-163``) regardless of what dtype its input arrives in.

``cast_forward_inputs=False`` matches torchtitan (``fsdp.py:226``): the data loader hands the
model int64 token ids, and an FSDP policy that helpfully casts *inputs* would try to cast them
too. The first module that needs a dtype change (the embedding lookup) produces bf16 by virtue
of the gathered bf16 weight.

What this policy does NOT do: it does not touch the optimizer. Master weights, Adam's first and
second moments, and the LR schedule are fp32 throughout — that is
:mod:`scratch_llm.training.t1_train`'s job, and it is the other half of "mixed" precision. See
``utils/mixed_precision.py`` for the accumulation experiment this policy is the production
answer to.
"""

from __future__ import annotations

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

# Name → dtype, spelled exactly as torchtitan's config spells it so the two configs can be
# compared as strings (``TORCH_DTYPE_MAP`` in torchtitan/config/__init__.py).
TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """``"bfloat16"`` → ``torch.bfloat16``. Raises on anything else, by name."""
    try:
        return TORCH_DTYPE_MAP[name]
    except KeyError:
        raise ValueError(
            f"unknown dtype {name!r}; expected one of {sorted(TORCH_DTYPE_MAP)}"
        ) from None


def bf16_mp_policy(
    param_dtype: str = "bfloat16", reduce_dtype: str = "float32"
) -> MixedPrecisionPolicy:
    """The rung's FSDP2 mixed-precision policy.

    Defaults are the ones the spec pins. ``reduce_dtype`` is refused if it is lower precision
    than ``param_dtype``: that combination has no upside — it saves reduce-scatter bytes the
    all-gather already dwarfs — and every gradient in the run pays for it. Failing here is the
    cheapest place to catch a config that would otherwise only show up as a loss gap at step
    10 000.
    """
    p = resolve_dtype(param_dtype)
    r = resolve_dtype(reduce_dtype)
    if r in (torch.bfloat16, torch.float16) and p is torch.float32:
        raise ValueError(
            f"reduce_dtype={reduce_dtype} is lower precision than param_dtype={param_dtype}: "
            "the gradient reduction is the one place in a bf16 run that must not lose bits "
            "(see this module's docstring)"
        )
    return MixedPrecisionPolicy(param_dtype=p, reduce_dtype=r, cast_forward_inputs=False)


def autocast_dtype(param_dtype: str) -> torch.dtype | None:
    """The dtype for the *non-FSDP* half of the step.

    FSDP2's ``param_dtype`` casts parameters, not activations of ops that have no parameters
    (softmax, the SwiGLU product, RoPE). ``torch.autocast`` covers those with the per-op policy
    described in ``utils/mixed_precision.py``. Returns ``None`` for fp32 so the caller can use a
    ``nullcontext`` and keep the fp32 path byte-identical to a run with no autocast at all.
    """
    d = resolve_dtype(param_dtype)
    return None if d is torch.float32 else d


__all__ = ["TORCH_DTYPE_MAP", "autocast_dtype", "bf16_mp_policy", "resolve_dtype"]
