"""K1/H-R4 — persistent, cluster-of-2, TMA-multicast warpgroup-MMA GEMM (Hopper, sm_90a).

The top Hopper rung. Everything H-R3 has (TMA + mbarrier pipeline, warp specialisation, 128x256
tiles) plus the four things that only make sense once the grid stops tracking the problem:

* **Persistent** — the grid is the SM count, not ``ceil(M/128) * ceil(N/256)``. A CTA is launched
  once and walks a strided sequence of output tiles, so the pipeline prologue is paid once per SM
  (132 times at 4096^3) instead of once per tile (512 times).
* **Tile scheduler** — because the grid no longer encodes the tile, the ORDER becomes a variable.
  A grouped raster walks a ``GROUP_M x GROUP_N`` supertile of cluster-tiles before moving on, so
  the A-rows and B-columns a group touches stay resident in L2.
* **Cluster of 2** — two CTAs launched together on adjacent SMs, splitting ``TILE_M`` and sharing
  one ``TILE_N`` column strip.
* **TMA multicast** — which is what makes the cluster pay for itself: the shared B tile is fetched
  from HBM once and landed in both CTAs' shared memory.

What the grid buys is *not* a smaller ``ceil``: for equal-cost tiles the time is still
``ceil(tiles / clusters)`` tile-times, and closing that last gap is stream-K's job, not
persistence's (plan §05 K1 lists it as optional for this rung). What persistence removes is the
*launch's* dependence on the shape and the per-tile cold start. The CPU suite asserts exactly that
split and nothing wider — see ``tests/kernels/gemm/test_k1_h_r4.py``.

bf16, not fp16: the ladder's floor is cuBLAS **bf16** at 4096^3 (plan §04/§05), and a number
measured in one dtype against a floor in another is not a number.

Kernel source and the ``# HUY:`` hole: ``csrc/gemm/h_r4_persistent_bf16_sm90.cu``.
Spec, floor and kill rule: ``experiments/K1/H-R4/spec.md``. Map: ``experiments/K1/H-R4/map.md``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_arch
from scratch_llm.kernels.gemm.cuda._k1_loader import load_rung

#: The K1 rung identity, in one place, so the wrapper, the tests, the bench and the dry-dock
#: registry cannot drift apart.
RUNG = "K1/H-R4"
SOURCE = "h_r4_persistent_bf16_sm90.cu"
SYMBOL = "h_r4_persistent_bf16"
ARCH = (9, 0)

#: Tile shape and pipeline depth, mirrored from the ``.cu``'s defines. The Python side needs these
#: to check the shared-memory budget, the wave quantization and the TMA tensor maps without a GPU
#: (``kernels.common.hopper_contracts``); the CPU contract tests assert the two agree.
TILE_M, TILE_N, TILE_K = 128, 256, 64
QSIZE = 3
NUM_CONSUMER_WG = 2
THREADS_PER_CTA = (NUM_CONSUMER_WG + 1) * 128

#: Cluster of 2 along M. Both CTAs own the same ``TILE_N`` column strip, which is why B — and only
#: B — is the tile worth multicasting.
CLUSTER_M, CLUSTER_N = 2, 1
CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

#: The grouped-raster supertile, in CLUSTER-tiles. From ``matmul_10.cuh:429``: upstream instantiates
#: ``Schedule<1, NUM_SM/CLUSTERS, BM*CLUSTER_M, BN*CLUSTER_N, 16/CLUSTER_M, 8/CLUSTER_N>`` — an 8x8
#: supertile of 256x256 cluster-tiles.
GROUP_M, GROUP_N = 8, 8

#: The warp-specialisation register split (``setmaxnreg``). Producer is one thread issuing TMA
#: descriptors; each consumer holds 128 fp32 accumulators.
PRODUCER_REGS, CONSUMER_REGS = 24, 240

#: Shared-memory descriptor offsets in bytes, per operand layout. A is K-major (torch ``[M,K]``
#: row-major); B is MN-major (torch ``[K,N]`` row-major, so N-contiguous, so the ``trans_b = 1``
#: form). See the ``.cu`` header for why B cannot be K-major here and what it costs.
A_DESC_LBO_BYTES, A_DESC_SBO_BYTES = 16, 1024
B_DESC_LBO_BYTES, B_DESC_SBO_BYTES = TILE_K * 128, 1024

#: 128 B / sizeof(bf16). The innermost dimension of every TMA box at this rung is exactly one
#: 128 B swizzle atom, which is what forces the rank-3 tensor maps — and what makes ``N % 64`` a
#: hard legality constraint rather than a tuning preference.
TMA_ATOM_ELEMS = 64


def stage_bytes() -> int:
    """Shared memory for one pipeline stage: an A tile plus a B tile, both bf16."""
    return (TILE_M * TILE_K + TILE_K * TILE_N) * 2


def barrier_bytes() -> int:
    """The ``full``/``empty`` mbarrier arrays — 8 bytes each, two arrays of ``QSIZE``."""
    return 2 * QSIZE * 8


def cluster_tiles(m: int, n: int) -> tuple[int, int]:
    """The problem in CLUSTER-tiles: what the scheduler indexes, ``(tiles_m, tiles_n)``.

    A cluster-tile is ``(TILE_M * CLUSTER_M) x (TILE_N * CLUSTER_N)`` because the two CTAs of a
    cluster cover adjacent ``TILE_M`` strips of one ``TILE_N`` column.
    """
    return (
        math.ceil(m / (TILE_M * CLUSTER_M)),
        math.ceil(n / (TILE_N * CLUSTER_N)),
    )


def persistent_grid(sm_count: int) -> int:
    """CTAs the launcher asks for: one per SM, rounded down to a whole number of clusters.

    The whole content of the word *persistent* is that this does not take ``m`` or ``n``. The
    launcher computes the same expression from ``cudaDevAttrMultiProcessorCount``; the CPU suite
    asserts the two agree and that the answer is the same at every spec shape.
    """
    if sm_count < CLUSTER_SIZE:
        raise ValueError(
            f"persistent_grid: {sm_count} SMs is fewer than one cluster of {CLUSTER_SIZE}"
        )
    return (sm_count // CLUSTER_SIZE) * CLUSTER_SIZE


def shape_is_legal(m: int, n: int, k: int) -> list[str]:
    """Why this rung refuses a shape, as a list of reasons (empty == legal).

    Both entries are TMA constraints, not tile-shape preferences. The tensor map's innermost box
    dimension *is* the 128 B swizzle atom, so each operand's contiguous extent must be a whole
    number of atoms: ``K`` for A (``[M,K]`` row-major) and ``N`` for B (``[K,N]`` row-major).
    ``K % TILE_K`` is the stricter of the two K rules and subsumes ``K % 64``.
    """
    errs: list[str] = []
    if k % TILE_K:
        errs.append(
            f"K={k} is not a multiple of TILE_K={TILE_K} (no K-remainder path at this rung)"
        )
    if n % TMA_ATOM_ELEMS:
        errs.append(
            f"N={n} is not a multiple of {TMA_ATOM_ELEMS} — the B tensor map's innermost box is "
            f"one 128 B swizzle atom wide, and TMA cannot describe a partial atom"
        )
    return errs


def h_r4_gemm(a: Tensor, b: Tensor) -> Tensor:
    """``C = A @ B`` for bf16 ``a`` [M,K] and ``b`` [K,N]; fp32-accumulated, **fp32 output**.

    fp32 out, not bf16: the accumulator is fp32 in registers and rounding it on the way out would
    fold the kernel's error together with the output cast's, leaving the correctness gate unable to
    tell a scheduler bug from a rounding difference. It also keeps this rung's error directly
    comparable with H-R1's, which is the point of one ladder.

    ``M`` is predicated (a cluster whose ``rank_m`` lands past the last tile row still participates
    in the multicast and writes nothing). ``K`` and ``N`` are not: see :func:`shape_is_legal`.

    Raises ``RuntimeError`` on a non-Hopper device and
    :class:`~scratch_llm.kernels.gemm.cuda._k1_loader.HoleOpenError` while the persistent loop is
    still an unfilled hole.
    """
    require_arch(ARCH, fn_name="h_r4_gemm")
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError(
            f"h_r4_gemm expects bfloat16 operands (wgmma.f32.bf16.bf16), got {a.dtype} and "
            f"{b.dtype}; the K1 floor is cuBLAS bf16, so another dtype measures against no floor"
        )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"h_r4_gemm expects 2-D operands, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"h_r4_gemm inner dimensions disagree: {a.shape[1]} vs {b.shape[0]}")
    errs = shape_is_legal(a.shape[0], b.shape[1], a.shape[1])
    if errs:
        raise ValueError(f"h_r4_gemm cannot run this shape: {'; '.join(errs)}")
    return getattr(load_rung("k1_h_r4", SOURCE, SYMBOL, ARCH), SYMBOL)(a, b)


def reference_gemm(a: Tensor, b: Tensor) -> Tensor:
    """The oracle: the same bf16 inputs, multiplied in fp32.

    Slow, obviously correct, and — importantly — *not* a different problem. It up-casts the
    identical bf16 operands the kernel receives, so the only difference between this and the kernel
    is accumulation order and the tensor core's internal rounding. Comparing against an fp32 *input*
    GEMM instead would fold the operands' own quantization error into the tolerance and hide a real
    kernel bug inside it.
    """
    return torch.matmul(a.float(), b.float())
