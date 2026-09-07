"""K1/B-R6 — NVFP4 block-scaled GEMM, ``mma.sync`` path (Blackwell client, sm_120a).

The last K1 rung: the operands stop being a dtype and become a *format*. Each 16 K-elements of A
and of B carry one UE4M3 scale, the elements themselves are E2M1 — four bits, one mantissa bit,
eight magnitudes — and the tensor core multiplies scaled operands directly, accumulating in fp32
(``mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64``,
``cute/arch/mma_sm120.hpp:3216``). The plan's exit for this rung is ``<= 2x SoL``, not a percentage
of cuBLAS, because there is no bf16 kernel doing the same arithmetic to compare against.

Two silicon paths, one layout. B200 issues the same arithmetic as ``tcgen05.mma…kind::mxf4nvf4``
with the scales in TMEM, reached through B-R5's CuTe DSL module; sm120 has no TMEM and issues it
per warp with the scales in one ordinary 32-bit register. The *global-memory* scale layout is
byte-identical between them (``sm120_blockscaled_mma_builder.inl:164`` reuses
``Sm1xxBlockScaledConfig``), which is why :mod:`scratch_llm.kernels.gemm.nvfp4_layout` is arch-free
and why this rung's compilable artifact being the sm120 one costs nothing.

**The oracle is what makes this rung's tolerance derivable.** :func:`reference_gemm` quantizes,
dequantizes, and multiplies in fp32 — so it solves the same problem the kernel solves, and the only
difference between them is accumulation order and the fragment layout. :func:`quantization_error`
measures the *other* residual, the one the format costs against an fp32 GEMM on the original
operands. Huy's tolerance has to separate the two: a bound that absorbs the format error cannot
fail on a wrong fragment mapping.

Kernel source and the ``// HUY:`` hole: ``csrc/gemm/b_r6_nvfp4_sm120.cu``.
Spec, floor and kill rule: ``experiments/K1/B-R6/spec.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_arch
from scratch_llm.kernels.gemm.cuda._k1_loader import load_rung
from scratch_llm.kernels.gemm.nvfp4_layout import (
    ATOM_BYTES,
    BLK_SF,
    E2M1_MAGNITUDES,
    E2M1_MAX,
    FP4_E2M1,
    SF_UE4M3,
    SF_VEC_SIZE,
    sf_atom_grid,
    sf_tensor_bytes,
)

#: The K1 rung identity, in one place, so the wrapper, the tests, the bench and the dry-dock
#: registry cannot drift apart.
RUNG = "K1/B-R6"
SOURCE = "b_r6_nvfp4_sm120.cu"
SYMBOL = "b_r6_nvfp4_mma_sync"
ARCH = (12, 0)

#: Tile shape, mirrored from the ``BM``/``BN``/``BK`` defines in the ``.cu``; the CPU contract tests
#: assert the two agree. ``TILE_K = 64`` is not free — it is ``BLK_SF * SF_VEC_SIZE``, one scale
#: atom's K extent, and simultaneously the mma's ``k``, so one mainloop step consumes exactly one
#: 512-byte scale block per operand.
TILE_M, TILE_N, TILE_K = 128, 128, 64
THREADS_PER_CTA = 256


@dataclass(frozen=True)
class NVFP4Operand:
    """One quantized operand: packed E2M1 elements plus its swizzled UE4M3 scale tensor.

    ``packed`` is ``[rows, k // 2]`` uint8 with the LOW nibble holding the lower ``k`` — the order
    a 32-bit register hands to the MMA, lowest element in the lowest bits. ``scales`` is a flat
    uint8 tensor in the layout :mod:`~scratch_llm.kernels.gemm.nvfp4_layout` describes, NOT a
    ``[rows, k // 16]`` array: the permutation is the point, and keeping the quantizer's output in
    the hardware's layout is what stops a "just reshape it" from appearing later in the kernel.

    ``rows`` and ``k`` are the LOGICAL shape, carried because neither tensor's shape recovers ``k``
    unambiguously once it is packed and the scales are padded to whole atoms.
    """

    packed: Tensor
    scales: Tensor
    rows: int
    k: int

    def __post_init__(self) -> None:
        if self.packed.shape != (self.rows, self.k // 2):
            raise ValueError(
                f"packed has shape {tuple(self.packed.shape)}, expected ({self.rows}, {self.k // 2})"
            )
        if self.scales.numel() != sf_tensor_bytes(self.rows, self.k):
            raise ValueError(
                f"scales has {self.scales.numel()} bytes, the swizzled layout for "
                f"{self.rows}x{self.k} needs {sf_tensor_bytes(self.rows, self.k)}"
            )


# =============================================================================================
# Lookup tables, built once per device from the format definitions (never typed by hand)
# =============================================================================================


@lru_cache(maxsize=8)
def _e2m1_table(device: str) -> Tensor:
    """Code (0..15) -> value, fp32. Signed, so the sign bit is decoded by the gather itself."""
    return torch.tensor([FP4_E2M1.decode(c) for c in range(16)], dtype=torch.float32, device=device)


@lru_cache(maxsize=8)
def _ue4m3_table(device: str) -> Tensor:
    """Code (0..127) -> value, fp64. The NaN code (0x7F) holds ``inf``; see ``FpFormat.values``."""
    return torch.tensor(SF_UE4M3.values(), dtype=torch.float64, device=device)


@lru_cache(maxsize=8)
def _ue4m3_ascending(device: str) -> tuple[Tensor, Tensor]:
    """``(sorted finite values, their codes)`` — the grid a block scale is rounded UP onto."""
    pairs = sorted((v, c) for c, v in enumerate(SF_UE4M3.values()) if v != float("inf"))
    values = torch.tensor([v for v, _ in pairs], dtype=torch.float64, device=device)
    codes = torch.tensor([c for _, c in pairs], dtype=torch.uint8, device=device)
    return values, codes


@lru_cache(maxsize=8)
def _e2m1_midpoints(device: str) -> Tensor:
    """The seven midpoints between consecutive E2M1 magnitudes — the round-to-nearest boundaries."""
    mags = E2M1_MAGNITUDES
    return torch.tensor(
        [0.5 * (mags[i] + mags[i + 1]) for i in range(7)], dtype=torch.float64, device=device
    )


@lru_cache(maxsize=64)
def _sf_offset_table(rows: int, k: int, device: str) -> Tensor:
    """``[rows, k // 16]`` int64: the byte offset of every block's scale, vectorized.

    The vectorized twin of :func:`~scratch_llm.kernels.gemm.nvfp4_layout.sf_byte_offset`; a CPU
    test asserts they agree element by element, which is the only thing that makes the fast path
    trustworthy. Cached because it depends only on the shape, and a 4096x4096 operand rebuilds a
    1M-element index tensor on every call otherwise.
    """
    _, k_atoms = sf_atom_grid(rows, k)
    row = torch.arange(rows, dtype=torch.int64, device=device)
    blk = torch.arange(k // SF_VEC_SIZE, dtype=torch.int64, device=device)
    m_atom, row_in = row // 128, row % 128
    r0, r1 = row_in % 32, row_in // 32
    k_atom, k1 = blk // BLK_SF, blk % BLK_SF
    atom = m_atom[:, None] * k_atoms + k_atom[None, :]
    return ATOM_BYTES * atom + (16 * r0 + 4 * r1)[:, None] + k1[None, :]


# =============================================================================================
# The quantization oracle
# =============================================================================================


def quantize_nvfp4(x: Tensor) -> NVFP4Operand:
    """``[rows, k]`` real -> NVFP4, blocks along the LAST dimension (which must be the contraction).

    Per block of :data:`~scratch_llm.kernels.gemm.nvfp4_layout.SF_VEC_SIZE` elements:
    ``scale = ceil_ue4m3(amax / 6)``, ``code = round_to_nearest_even_e2m1(x / scale)``. The scale
    rounds **up**, which is what makes ``|x / scale| <= 6`` a theorem rather than a hope — see
    :func:`~scratch_llm.kernels.gemm.nvfp4_layout.scale_for_amax` for the algebra and for the
    second property (``amax / scale >= 5.33``, so the block's largest element reaches 6.0) that
    makes quantize-dequantize-quantize a fixed point.

    The division is done in float64. Not caution: E2M1's midpoints are 0.25, 0.75, ..., 5.0, real
    blocks land exactly on them, and a float32 quotient that misses a midpoint by one ulp turns a
    ties-to-even decision into a coin flip — which shows up later as a round-trip error bound that
    holds for most seeds.

    Raises on a non-finite input: E2M1 encodes neither NaN nor Inf (``NanInfEncoding::NONE``), so
    there is no honest thing to store. Silently saturating a NaN to 6.0 would let a poisoned
    activation reach a kernel as a large finite number, which is the failure mode the K1 suite's
    NaN test exists to prevent on every other rung.
    """
    if x.ndim != 2:
        raise ValueError(f"quantize_nvfp4 expects a 2-D operand, got {x.ndim}-D")
    rows, k = int(x.shape[0]), int(x.shape[1])
    if k % SF_VEC_SIZE:
        raise ValueError(
            f"quantize_nvfp4: K={k} is not a multiple of SF_VEC_SIZE={SF_VEC_SIZE}; a partial "
            f"scale block has no encoding"
        )
    dev = str(x.device)
    xf = x.to(torch.float64)
    if not torch.isfinite(xf).all():
        raise ValueError(
            "quantize_nvfp4: operand contains NaN or Inf, and E2M1 encodes neither "
            "(NanInfEncoding::NONE) — clean the tensor before quantizing rather than letting the "
            "format saturate it to a finite 6.0"
        )

    blocks = xf.view(rows, k // SF_VEC_SIZE, SF_VEC_SIZE)
    amax = blocks.abs().amax(dim=-1)

    grid, codes = _ue4m3_ascending(dev)
    idx = torch.searchsorted(grid, (amax / E2M1_MAX).contiguous(), right=False)
    if bool((idx >= grid.numel()).any()):
        worst = float(amax.max())
        raise ValueError(
            f"quantize_nvfp4: a block's amax/{E2M1_MAX:g} = {worst / E2M1_MAX:g} exceeds UE4M3's "
            f"maximum {float(grid[-1]):g}; the scale is unrepresentable — rescale the operand"
        )
    sf_code = codes[idx]
    scale = _ue4m3_table(dev)[idx.to(torch.int64)]

    # scale == 0 only for an all-zero block, where every quotient is 0/0; divide by 1 there and let
    # the zeros fall out on their own rather than filtering NaNs back out afterwards.
    y = blocks / torch.where(scale > 0, scale, torch.ones_like(scale)).unsqueeze(-1)

    mid = _e2m1_midpoints(dev)
    a = y.abs()
    lo = torch.bucketize(a, mid, right=False)  # ties round DOWN
    hi = torch.bucketize(a, mid, right=True)  # ties round UP
    # The magnitude index IS the (exponent, mantissa) code, so an even index has mantissa 0. A tie
    # at midpoint j is between indices j and j+1; the even one wins, which is j+1 exactly when j is
    # odd. Non-ties have lo == hi and the choice is vacuous.
    mag_idx = torch.where((hi > lo) & (lo % 2 == 1), hi, lo)
    # Zero is emitted canonically (code 0x0), never as E2M1's negative zero 0x8. The two are
    # numerically identical and the MMA cannot tell them apart, but keeping the sign of a value
    # that has just been rounded away makes the packed bytes depend on data that no longer exists
    # in them — and that is exactly what breaks bit-level quantize-dequantize-quantize idempotence,
    # which is the cheapest end-to-end check this quantizer has.
    code = mag_idx.to(torch.uint8) | torch.where((y < 0) & (mag_idx > 0), 8, 0).to(torch.uint8)

    flat = code.view(rows, k)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)

    scales = torch.zeros(sf_tensor_bytes(rows, k), dtype=torch.uint8, device=x.device)
    scales[_sf_offset_table(rows, k, dev).reshape(-1)] = sf_code.reshape(-1)
    return NVFP4Operand(packed=packed.contiguous(), scales=scales, rows=rows, k=k)


def dequantize_nvfp4(op: NVFP4Operand) -> Tensor:
    """NVFP4 -> ``[rows, k]`` fp32. Exact: every ``scale * magnitude`` is an fp32 number.

    The inverse of :func:`quantize_nvfp4` in the only sense a lossy format allows — it recovers the
    values the tensor core will actually multiply, which is precisely what an oracle needs.
    """
    dev = str(op.packed.device)
    lo = op.packed & 0xF
    hi = op.packed >> 4
    codes = torch.stack((lo, hi), dim=-1).view(op.rows, op.k)  # low nibble first == lower k
    values = _e2m1_table(dev)[codes.to(torch.int64)]
    scale = _ue4m3_table(dev)[op.scales[_sf_offset_table(op.rows, op.k, dev)].to(torch.int64)]
    return (
        values.view(op.rows, op.k // SF_VEC_SIZE, SF_VEC_SIZE) * scale.unsqueeze(-1).float()
    ).view(op.rows, op.k)


# =============================================================================================
# The kernel
# =============================================================================================


def b_r6_gemm_packed(a: NVFP4Operand, b: NVFP4Operand) -> Tensor:
    """``C = A @ B.T`` for two already-quantized operands; fp32-accumulated, **fp32 output**.

    ``b`` is ``[N, K]`` — the column-major ``[K, N]`` operand the ``.row.col`` MMA wants, which is
    the same bytes (upstream: A RowMajor, B ColumnMajor,
    ``79a_blackwell_geforce_nvfp4_bf16_gemm.cu:97,102``).

    This is the entry a performance number should be taken through. Quantization is
    ``O(MK + KN)`` and memory bound; at 4096^3 it is a double-digit percentage of the GEMM, and
    folding it into a window whose FLOP model is ``2*M*N*K`` produces a TFLOP/s that describes
    neither operation.

    Raises ``RuntimeError`` on a non-sm120 device and
    :class:`~scratch_llm.kernels.gemm.cuda._k1_loader.HoleOpenError` while the kernel body is still
    an unfilled hole.
    """
    require_arch(ARCH, fn_name="b_r6_gemm_packed")
    if a.k != b.k:
        raise ValueError(f"b_r6_gemm_packed inner dimensions disagree: {a.k} vs {b.k}")
    if a.k % TILE_K:
        raise ValueError(
            f"b_r6_gemm_packed requires K % {TILE_K} == 0 at this rung (one mainloop step is one "
            f"scale atom and one mma k; no remainder path); got K={a.k}"
        )
    fn = getattr(load_rung("k1_b_r6", SOURCE, SYMBOL, ARCH), SYMBOL)
    return fn(a.packed, a.scales, b.packed, b.scales)


def b_r6_gemm(a: Tensor, b: Tensor) -> Tensor:
    """``C = A @ B`` for real ``a`` ``[M,K]`` and ``b`` ``[K,N]``, quantizing both to NVFP4 first.

    The convenience entry, and the one ``bench/kernels/gemm/k1_ladder.py`` reaches (its registry
    calls ``fn(a, b)`` with bf16 operands). **The quantization happens inside this call**, so a
    timing loop around it measures a GEMM plus two memory-bound quantize passes; take the rung's
    number through :func:`b_r6_gemm_packed` with the operands quantized outside the window, and
    read the spec's Measurement line before quoting anything measured this way.
    """
    require_arch(ARCH, fn_name="b_r6_gemm")
    if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(
            f"b_r6_gemm quantizes its operands to NVFP4, so it takes a real dtype it can round "
            f"(bfloat16 or float32), got {a.dtype} and {b.dtype}. Already-quantized operands go "
            f"to b_r6_gemm_packed"
        )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"b_r6_gemm expects 2-D operands, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"b_r6_gemm inner dimensions disagree: {a.shape[1]} vs {b.shape[0]}")
    return b_r6_gemm_packed(quantize_nvfp4(a), quantize_nvfp4(b.t().contiguous()))


# =============================================================================================
# The oracles
# =============================================================================================


def reference_gemm(a: Tensor, b: Tensor) -> Tensor:
    """The correctness oracle: quantize both operands, dequantize, multiply in fp32.

    It solves the SAME problem the kernel solves. That is the whole design: the kernel's operands
    are the E2M1 values this function dequantizes, so the only differences left are accumulation
    order and the register-fragment mapping — the two things the correctness gate is actually
    testing. An oracle that multiplied the original bf16 operands instead would fold the format's
    own error into the tolerance, and a wrong fragment layout could hide underneath it.

    The scale is applied to the OPERAND, not to the product, matching the upstream reference
    (``tools/util/include/cutlass/util/reference/host/gett.hpp:555-560`` multiplies by ``SfA`` in
    fp32 before forming the product) and matching what the hardware does.
    """
    qa = quantize_nvfp4(a.float() if a.dtype is not torch.float32 else a)
    qb = quantize_nvfp4(
        b.t().contiguous().float() if b.dtype is not torch.float32 else b.t().contiguous()
    )
    return torch.matmul(dequantize_nvfp4(qa), dequantize_nvfp4(qb).t())


def quantization_error(a: Tensor, b: Tensor) -> dict[str, float]:
    """How much the FORMAT costs, measured against an fp32 GEMM on the original operands.

    Returns ``max_abs``, ``rel_max`` (normalised by ``max|C_fp32|``) and ``rel_fro``. This is not
    the correctness gate and must never be used as one — it is the quantity the gate's tolerance
    has to be argued *against*: a bound larger than this is a bound a wrong kernel passes.

    The separation is the rung's numerics lesson. ``|C_kernel - C_fp32|`` has two independent
    parts: the format residual this function measures, which is a property of E2M1 and UE4M3 and
    of the data, and the kernel residual, which is fp32 accumulation order over K. Only the second
    belongs in ``TOL_CONST``.
    """
    ref = reference_gemm(a, b)
    exact = torch.matmul(a.float(), b.float())
    delta = (ref - exact).abs()
    scale = exact.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs": float(delta.max()),
        "rel_max": float(delta.max() / scale),
        "rel_fro": float(delta.norm() / exact.norm().clamp_min(torch.finfo(torch.float32).tiny)),
    }
