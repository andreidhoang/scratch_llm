"""Hopper FlashAttention-3 (sm_90a) — the warp-specialized prefill frontier rung.

FA3 is the rung FA2 (``attention/prefill/fa2.py``, the Triton online-softmax
kernel) climbs to on Hopper. The load-bearing structural lifts over FA2:

  1. **Warp specialization** — a dedicated *producer* warpgroup issues TMA loads
     while *consumer* warpgroups do the WGMMA math. FA2 is a single-warpgroup
     kernel that serializes load and compute; FA3 overlaps them via mbarriers.
  2. **TMA (tensor-memory accelerator)** — asynchronous bulk loads of Q/K/V tiles
     straight into SMEM, no per-thread global load. Paired with WGMMA's
     SMEM-operand model, this is the Hopper data-movement story.
  3. **WGMMA for both QK^T and softmax(P)V** — FA2 uses Triton ``tl.dot`` which
     maps to mma.sync (Ampere/Hopper base ISA). FA3 issues ``wgmma.mma_async``
     against hand-encoded SMEM descriptors (the same machinery as
     ``kernels.gemm.cuda.wgmma``).
  4. **Ping-pong SMEM double-buffering** — while consumer 0 drains tile N to the
     softmax, consumer 1 starts the WGMMA for tile N+1. The softmax (the
     memory-bound op) is *hidden behind* the WGMMA (the compute-bound op).

ISA gate: WGMMA + TMA + mbarrier all assemble only under ``-arch=sm_90a``.
This loader enforces ``require_cc(9, 0)`` and hardcodes the JIT arch to sm_90a.

CORRECTNESS HONESTY: the kernel body in ``csrc/attention/fa3_hopper.cu`` is a
compile-gated structural skeleton (promoted from ``performance/rental/kernels/``).
Runtime correctness is deferred to the H100/H200 rental day; the oracle test
(``tests/kernels/test_flash_attention_fa3.py``) compares against
``flash_attention_forward`` (the CPU FA2 oracle) there.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_cc

_CU = Path(__file__).resolve().parents[5] / "csrc" / "attention" / "fa3_hopper.cu"


@lru_cache(maxsize=1)
def _module():  # noqa: ANN202 - opaque pybind extension module
    # 1. AOT path.
    try:
        from scratch_llm import _scratch_llm_kernels as _ext  # noqa: PLC0415

        if hasattr(_ext, "flash_attention_fa3_forward"):
            return _ext
    except ImportError:
        pass

    # 2. JIT fallback — FA3 needs sm_90a (WGMMA + TMA + mbarrier).
    from torch.utils.cpp_extension import load

    return load(
        name="flash_attention_fa3",
        sources=[str(_CU)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90a"],
        verbose=False,
    )


def flash_attention_fa3_forward(
    q: Tensor, k: Tensor, v: Tensor, *, is_causal: bool = True
) -> Tensor:
    """FlashAttention-3 forward on Hopper.

    Args:
        q, k, v: ``[..., S, D]`` float16; ``D`` must be 64 (one WGMMA atom width),
            leading dims identical across Q/K/V (self-attention shape for now).
        is_causal: apply the causal mask to the lower triangle.

    Returns ``O`` of shape ``[..., S, D]`` float16.
    """
    require_cc(9, 0, fn_name="flash_attention_fa3_forward")
    if not (q.dtype == k.dtype == v.dtype == torch.float16):
        raise TypeError("flash_attention_fa3_forward expects float16 inputs (wgmma.f16 operands)")
    return _module().flash_attention_fa3_forward(q, k, v, is_causal)
