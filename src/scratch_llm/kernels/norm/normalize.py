"""Triton RMSNorm + LayerNorm over the last dim of ``(M, N)`` — A2 Rung 3 (memory-bound norms).

Both kernels are the canonical whole-row-in-a-block reduction: one program per row loads its N
elements **once** from HBM into registers, reduces in fp32, normalizes the held tile, scales by the
(bf16) weight, and stores. Because the row is read exactly once and written once, the *ideal* HBM
traffic is ``2·M·N·2B`` (+ the O(N) weight/bias) for both — this is a bandwidth kernel, and at large
N it should approach the card's HBM peak.

The rung's question is **the cost of a second reduction**:

* ``rmsnorm_triton`` — ``x / sqrt(mean(x²) + eps) · w``: **one** reduction (sum of squares).
* ``layernorm_triton`` — ``(x − μ)/sqrt(σ² + eps) · w + b``: **two** reductions (μ, then σ² over the
  *same on-chip tile*). The second reduction is extra *compute*, not extra HBM traffic — both norms
  read x from HBM exactly once. So at large N (memory-bound) the two should measure nearly equal
  GB/s, and the one-reduction win is a small compute/latency edge visible mainly at small N. The
  bench quantifies exactly that.

Numerics: fp32 accumulation throughout (bf16 sum-of-squares would lose the tail). LayerNorm variance
is computed as ``mean((x−μ)²)`` (a genuine two-pass over the *held* tile), not the cancelling
``E[x²]−E[x]²`` form, so a single large outlier stays faithful to ``F.layer_norm``. Epsilon lives
under the sqrt, so a fully-zero row yields ``sqrt(eps) > 0`` — finite, no NaN.

Adversarial coverage (tests/test_norm.py, gpu): zero row (epsilon path), a single outlier, and N not
a multiple of the block (masked tail; ``BLOCK_N = next_pow2(N) ≥ N`` with a ``cols < N`` mask).

Invariant: matches ``F.rms_norm`` / ``F.layer_norm`` at rtol 1e-3 in bf16 on random + adversarial
inputs, on the GPU.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# Autotune only the launch geometry (warps/stages); BLOCK_N is next_pow2(N), passed per-shape as a
# constexpr so the whole row lives in one block and x is read from HBM exactly once. More warps hide
# HBM latency at large N; fewer avoid idle lanes at small N — the autotuner picks per N.
_CONFIGS = [triton.Config({}, num_warps=w, num_stages=s) for w in (1, 2, 4, 8, 16) for s in (1, 2)]


@triton.autotune(configs=_CONFIGS, key=["N"])
@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_xm,
    stride_ym,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)  # 1 HBM read

    ms = tl.sum(x * x, axis=0) / N  # the ONE reduction: mean of squares (fp32)
    rstd = 1.0 / tl.sqrt(ms + eps)  # eps under the sqrt → zero row = sqrt(eps), never NaN

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd) * w
    tl.store(y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_CONFIGS, key=["N"])
@triton.jit
def _layernorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    stride_xm,
    stride_ym,
    N,
    eps,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)  # 1 HBM read

    mean = tl.sum(x, axis=0) / N  # reduction 1: mean
    xc = tl.where(mask, x - mean, 0.0)  # masked tail must not pollute the variance sum
    var = tl.sum(xc * xc, axis=0) / N  # reduction 2: variance over the SAME held tile (no re-read)
    rstd = 1.0 / tl.sqrt(var + eps)  # zero row → var 0 → sqrt(eps), never NaN

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd) * w
    if HAS_BIAS:
        y += tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def _as_2d(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Flatten leading dims to a row-contiguous ``(M, N)`` view; return it plus the original shape."""
    n = x.shape[-1]
    x2 = x.reshape(-1, n).contiguous()
    return x2, tuple(x.shape)


def rmsnorm_triton(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """``x / sqrt(mean(x², last dim) + eps) · weight`` — matches :func:`F.rms_norm`.

    ``x``: ``(..., N)``; ``weight``: ``(N,)``. Returns x's dtype (fp32 reduction in-kernel).
    """
    x2, shape = _as_2d(x)
    m, n = x2.shape
    y = torch.empty_like(x2)
    _rmsnorm_kernel[(m,)](  # pyright: ignore[reportIndexIssue]  (triton autotuner launch grid)
        x2,
        weight.contiguous(),
        y,
        x2.stride(0),
        y.stride(0),
        n,
        eps,
        BLOCK_N=triton.next_power_of_2(n),
    )
    return y.reshape(shape)


def layernorm_triton(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, eps: float = 1e-5
) -> Tensor:
    """``(x − μ)/sqrt(σ² + eps) · weight + bias`` over the last dim — matches :func:`F.layer_norm`.

    ``x``: ``(..., N)``; ``weight``/``bias``: ``(N,)`` (bias optional). Returns x's dtype.
    """
    x2, shape = _as_2d(x)
    m, n = x2.shape
    y = torch.empty_like(x2)
    _layernorm_kernel[(m,)](  # pyright: ignore[reportIndexIssue]  (triton autotuner launch grid)
        x2,
        weight.contiguous(),
        bias.contiguous() if bias is not None else x2,  # unused ptr when HAS_BIAS=False
        y,
        x2.stride(0),
        y.stride(0),
        n,
        eps,
        HAS_BIAS=bias is not None,
        BLOCK_N=triton.next_power_of_2(n),
    )
    return y.reshape(shape)
