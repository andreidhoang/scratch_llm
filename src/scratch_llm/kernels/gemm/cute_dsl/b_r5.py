"""K1/B-R5 — tcgen05.mma into a TMEM accumulator, in the CuTe DSL (Blackwell DC, sm_100a).

The fifth rung of the K1 ladder, and the first that is not CUDA C++. The instrument changes with
the machine: on Blackwell the tensor core is programmed through ``tcgen05.mma`` (UMMA), one
instruction issued by ONE thread against operands in shared memory, and CUTLASS's own Blackwell
examples are written in the Python DSL rather than in C++. Writing the rung in the DSL is therefore
not a shortcut around CUDA — it is the code path NVIDIA ships — and what is being learned is the
Blackwell memory hierarchy, not another host-side build system.

**The structural difference from Hopper, which is the whole rung:**

  H-R1..H-R4 (wgmma): the accumulator lives in REGISTERS. The consumer warpgroup issues the MMA,
  the results land in its own registers, and the epilogue begins at "cast in registers".

  B-R5 (tcgen05): the accumulator lives in TENSOR MEMORY. TMEM is a separate per-SM SRAM —
  128 datapaths x 512 columns (``dense_gemm.py:599-602``) — addressed by the MMA instruction as
  its C operand (``dense_gemm.py:1009-1014``) and reachable from a warp only through an explicit
  ``tcgen05.ld``. So every output element crosses one extra hop before it can be written, and it
  does so on warps that are NOT the MMA warp, gated by its own pipeline (``acc_pipe``,
  ``dense_gemm.py:801-808``). The epilogue is TMEM -> RMEM -> SMEM -> GMEM
  (``dense_gemm.py:1064-1102``), not RMEM -> GMEM.

That has a budget consequence the tile shape must be chosen against, and it is decidable here with
no GPU: a 128x256 1-CTA accumulator costs ``cta_tile_n`` = 256 TMEM columns per stage
(``blackwell_helpers.py:2412-2418``), so the two accumulator stages this kernel runs
(``dense_gemm.py:118``) consume 512 columns — the ENTIRE TMEM. There is no headroom left to buy
mainloop/epilogue overlap by deepening that pipeline. ``cta_group::2`` halves the per-CTA M tile,
which drops the cost to 128 columns per stage (``blackwell_helpers.py:2402-2408``), and is the one
lever that buys the headroom back. That is why the plan row reads "1-CTA then cta_group::2": two
measurements of one tile, not two tunings.

Map, with file:line for every claim above: ``experiments/K1/B-R5/map.md``.
Spec, floor, kill rule: ``experiments/K1/B-R5/spec.md``.
The reference this rung climbs toward, at sha 59e3a333:
``oss/cutlass/examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm/dense_gemm.py``.

The ``# HUY:`` hole is the kernel body — the producer/MMA mainloop and the TMEM->RMEM->SMEM->GMEM
epilogue, at the end of ``B_R5DenseGemm.kernel``. Everything around it (arch gate, DSL import guard,
tile arithmetic, tensor plumbing, pipelines, warp roles, the oracle) is written and CPU-tested;
``tests/kernels/gemm/test_k1_b_r5.py`` is the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_arch
from scratch_llm.kernels.common.hopper_contracts import arch_for

#: The rung identity, in one place, so the wrapper, the tests and the bench cannot drift apart.
RUNG = "K1/B-R5"
#: Repo-relative path of the source carrying this rung's hole (``@pytest.mark.hole`` reads it).
SOURCE = "src/scratch_llm/kernels/gemm/cute_dsl/b_r5.py"
ARCH = (10, 0)

#: The DSL is not a dependency of this repo. It is a large CUDA-linked wheel that does nothing on a
#: Mac, so it is installed on the box only — and pinned exactly, because a JIT compiler's version
#: is part of the code it generates and therefore part of any number measured through it.
DSL_REQUIREMENT = "nvidia-cutlass-dsl==4.7.1"

# ---------------------------------------------------------------------------------------------
# Tile shape. From the plan row (§05 K1): 128x256x64, cta_group 1 then 2.
# ---------------------------------------------------------------------------------------------

#: The MMA tile, in the DSL's own units (``mn_tiler``). Upstream constrains M to 64 or 128 — the
#: instruction's native M — and its pytest sweeps exactly this (128, 256) with and without 2-CTA
#: (``test/examples/CuTeDSL/sm_100a/test_dense_gemm_persistent_prefetch.py:65-85``).
MMA_TILE_MN = (128, 256)

#: K comes out of the instruction, not out of taste: ``MmaF16BF16Op`` has instruction K = 16 for
#: bf16 (``blackwell_helpers.py:1195-1203``) and the example issues 4 of them per K-tile
#: (``dense_gemm.py:414``). 16 x 4 = 64 is the 64 in "128x256x64".
MMA_INST_K = 16
MMA_INST_TILE_K = 4
TILE_K = MMA_INST_K * MMA_INST_TILE_K

#: Accumulator stages in TMEM, and epilogue staging buffers in SMEM. Both are fixed upstream
#: (``dense_gemm.py:118``, ``:223-225``); the mainloop depth is not — see
#: :meth:`TileConfig.max_mainloop_stages`.
ACC_STAGE = 2
TMA_STORE_STAGE = 2

#: TMEM geometry, ``dense_gemm.py:599-602``. 512 columns is a hard ceiling the accumulator layout
#: is budgeted against, exactly as 227 KB is for shared memory.
TMEM_DATAPATHS = 128
TMEM_COLUMNS = 512

#: 6 warps, 192 threads (``dense_gemm.py:329``), specialised by role (``:826-831``). Warp 0 issues
#: the TMA store AND takes part in the epilogue, which is why the two roles overlap.
THREADS_PER_CTA = 192
WARP_ROLE_TMA_STORE = 0
WARP_ROLE_EPILOGUE = (0, 1, 2, 3)
WARP_ROLE_MMA = 4
WARP_ROLE_TMA_LOAD = 5

#: TMA requires 16 B alignment on the contiguous dimension of every operand. Upstream checks it on
#: the host before compiling anything (``blackwell_helpers.py:79``, ``:106-132``, ``:135-181``);
#: this rung checks it in the wrapper, because a shape that fails it is a shape this rung cannot
#: measure — not a kernel bug.
TMA_ALIGNMENT_BYTES = 16


@dataclass(frozen=True)
class TileConfig:
    """The tile shape's consequences, as integer arithmetic — decidable with no GPU.

    Every field below is derived by reading the reference, and the derivation is cited where it is
    used. This exists so that the shared-memory budget, the TMEM column budget and the epilogue
    sub-tile are settled (and asserted, in ``test_k1_b_r5.py``) before a B200 is rented, instead of
    being discovered as a launch failure or a silently-shallow pipeline on the box.
    """

    cta_group: int = 1
    mma_tile_mn: tuple[int, int] = MMA_TILE_MN
    tile_k: int = TILE_K
    acc_stage: int = ACC_STAGE
    ab_bits: int = 16  # bf16 operands: the K1 floor is bf16, so nothing else measures a floor
    d_bits: int = 32  # fp32 output — see b_r5_gemm's docstring for why the epilogue does not cast

    def __post_init__(self) -> None:
        if self.cta_group not in (1, 2):
            raise ValueError(f"cta_group must be 1 or 2 (CtaGroup.ONE/TWO), got {self.cta_group}")

    @property
    def use_2cta_instrs(self) -> bool:
        return self.cta_group == 2

    @property
    def cluster_shape_mn(self) -> tuple[int, int]:
        """(2, 1) for cta_group::2, else (1, 1) — the self-cast shape, ``dense_gemm.py:233-234``.

        Staying ON the self-cast shape keeps ``use_tma_multicast`` False: the CTA pair shares
        operands through the MMA itself, not through a multicast TMA. Multicast is a separate
        variable and belongs to a later measurement, not to this one.
        """
        return (2, 1) if self.use_2cta_instrs else (1, 1)

    @property
    def cta_tile_mnk(self) -> tuple[int, int, int]:
        """The tile ONE CTA owns: ``shape_div(mnk_tiler, (num_mma_ctas, 1, 1))``, ``:434``.

        Only M is divided. Under cta_group::2 the accumulator splits along M and each CTA still
        holds the full N (``dense_gemm.py:428-435``) — which is exactly why the TMEM column cost
        halves rather than quartering.
        """
        return (self.mma_tile_mn[0] // self.cta_group, self.mma_tile_mn[1], self.tile_k)

    @property
    def b_tile_nk(self) -> tuple[int, int]:
        """B's per-CTA SMEM tile: ``(cta_tile_n // num_mma_ctas, tile_k)``, ``dense_gemm.py:435``.

        B is divided a SECOND time. The accumulator keeps full N, but the operand each CTA stages
        does not — the peer CTA stages the other half. Missing this is how a 2-CTA shared-memory
        budget comes out 2x too large.
        """
        return (self.mma_tile_mn[1] // self.cta_group, self.tile_k)

    def smem_bytes_per_stage(self) -> int:
        """Shared memory one mainloop stage costs this CTA: its A tile plus its B tile."""
        cta_m = self.cta_tile_mnk[0]
        b_n = self.b_tile_nk[0]
        return (cta_m * self.tile_k + b_n * self.tile_k) * (self.ab_bits // 8)

    def epilogue_smem_bytes(self) -> int:
        """The TMA-store staging buffer — outside the mainloop pipeline, ``dense_gemm.py:582-589``."""
        epi_m, epi_n = self.epilogue_tile_mn
        return epi_m * epi_n * (self.d_bits // 8) * TMA_STORE_STAGE

    def max_mainloop_stages(self, *, smem_per_cta: int) -> int:
        """How deep the TMA->UMMA pipeline can be for a given per-CTA shared-memory ceiling.

        Upstream solves this on the host from a device query (``_compute_stages``,
        ``dense_gemm.py:88-157``, called ``:312``) and falls back to 4 (``:223-225``). This rung
        computes it in Python from the same two numbers instead, so the depth is a static constant
        the CPU tests can assert rather than something the first launch reveals. It is a CEILING:
        mbarrier bytes and the DSL's own scratch are not modelled, so a real build may legitimately
        come out one stage shallower.
        """
        room = smem_per_cta - self.epilogue_smem_bytes()
        return max(0, room // self.smem_bytes_per_stage())

    @property
    def acc_tmem_columns_per_stage(self) -> int:
        """TMEM columns one accumulator stage costs, ``blackwell_helpers.py:2402-2418``.

        1-CTA at M=128 is NonInterleaved — one column per N value, so the cost is ``cta_tile_n``.
        Under cta_group::2 each CTA's 128 datapaths hold only ``cta_tile_m`` = 64 rows, so
        ``128 / cta_tile_m`` = 2 N-values share a column and the cost halves.
        """
        cta_m, cta_n, _ = self.cta_tile_mnk
        if self.use_2cta_instrs:
            return cta_n // (TMEM_DATAPATHS // cta_m)
        # M == 128 is NonInterleaved. The interleaved 1-CTA M == 64 case (bh:2409-2417) is a
        # different tile and not this rung's.
        return cta_n

    @property
    def acc_tmem_columns(self) -> int:
        """What the accumulator pipeline holds in total. Must not exceed :data:`TMEM_COLUMNS`."""
        return self.acc_tmem_columns_per_stage * self.acc_stage

    @property
    def epilogue_tile_mn(self) -> tuple[int, int]:
        """The epilogue sub-tile, as ``compute_epilogue_tile_shape`` derives it, ``bh:2265-2331``.

        Restricted to this rung's case — no source C (the GEMM is D = A@B, not alpha*A@B + beta*C)
        and an N-contiguous D — with each step carrying the line it comes from. For 128x256 with an
        fp32 D this lands on (128, 32): eight epilogue sub-tiles per CTA tile, i.e. eight
        TMEM->RMEM->SMEM->GMEM trips per output tile.
        """
        cta_m, cta_n, _ = self.cta_tile_mnk
        warp_m, warp_n = (
            (2, 2) if (cta_m == 64 and self.use_2cta_instrs) else (4, 1)
        )  # bh:2272-2275
        tile_m = min(cta_m, 32 * warp_m)  # bh:2285-2287, 32 datapaths per subpartition
        n_perf = (8192 if self.d_bits == 4 else 4096) // tile_m  # bh:2293-2295, source disabled
        while cta_n % n_perf:  # bh:2308-2309, no ragged last sub-tile
            n_perf //= 2
        n_min_d = (128 // self.d_bits) * warp_n  # bh:2316-2321, D N-major => 128-bit store granule
        n_min_c = 8 * warp_n  # bh:2322-2323, source disabled
        tile_n = min(cta_n, max(n_perf, n_min_c, n_min_d))  # bh:2326
        if cta_n % tile_n:  # bh:2327-2329
            tile_n = cta_n
        return (tile_m, tile_n)


def tile_config(cta_group: int = 1) -> TileConfig:
    """This rung's tile at ``cta_group`` 1 or 2 — the only two configurations the spec measures."""
    return TileConfig(cta_group=cta_group)


def tma_alignment_violations(m: int, n: int, k: int, *, cfg: TileConfig | None = None) -> list[str]:
    """TMA contiguous-dimension alignment for this wrapper's layouts (empty == legal).

    Mirrors ``check_gemm_tma_alignment`` / ``check_tma_tensor_alignment``
    (``blackwell_helpers.py:106-181``): the contiguous dimension of every TMA operand must be a
    multiple of ``16 B / element``. With ``a`` [M,K] and ``b`` [K,N] both row-major and an fp32
    row-major output, the contiguous dimensions are K for A (bf16 -> K % 8), N for B (bf16, seen as
    an N-major (N,K) -> N % 8) and N for D (fp32 -> N % 4).

    This is a property of the SHAPE, not of the kernel: no amount of correct kernel writing makes
    an unaligned shape TMA-addressable. The alternative — ``.contiguous()`` on the way in — is a
    device-to-device copy inside the timed region, which is a different measurement.
    """
    cfg = cfg or tile_config()
    ab_elems = TMA_ALIGNMENT_BYTES * 8 // cfg.ab_bits
    d_elems = TMA_ALIGNMENT_BYTES * 8 // cfg.d_bits
    errs: list[str] = []
    if k % ab_elems:
        errs.append(f"A TMA load: K={k} must be a multiple of {ab_elems} (bf16, K-contiguous)")
    if n % ab_elems:
        errs.append(f"B TMA load: N={n} must be a multiple of {ab_elems} (bf16, N-contiguous)")
    if n % d_elems:
        errs.append(f"D TMA store: N={n} must be a multiple of {d_elems} (fp32, N-contiguous)")
    return errs


def mainloop_stages(cta_group: int = 1) -> int:
    """The static mainloop depth this rung launches with, for ``cta_group`` 1 or 2."""
    return tile_config(cta_group).max_mainloop_stages(smem_per_cta=arch_for(ARCH).smem_per_cta)


@lru_cache(maxsize=1)
def require_cute_dsl() -> SimpleNamespace:
    """Import the CuTe DSL, or raise an ``ImportError`` that says exactly what to install.

    Lazy, and deliberately so: this module must IMPORT on a laptop with no CUDA and no DSL, or
    pytest cannot collect the contract tests that are the whole reason the rung is written before
    the rental window opens. The names bound here are the ones ``dense_gemm.py:34-40`` binds, so
    the kernel below reads against the reference line for line.
    """
    try:
        import cutlass
        import cutlass.utils.blackwell_helpers as sm100_utils
        from cutlass import cute
        from cutlass.cute import experimental as cute_ext
        from cutlass.cute.runtime import from_dlpack
    except ImportError as exc:
        raise ImportError(
            f"{RUNG} needs the CuTe DSL, which is not installed. On the B200 box:\n"
            f"    uv pip install {DSL_REQUIREMENT}\n"
            "It is not a dependency of this repo: the wheel is CUDA-linked and JIT-compiles for "
            "sm_100a, so it does nothing on a machine without a Blackwell datacenter part. Pin the "
            "version — a JIT compiler's version is part of the code it generates, and therefore "
            "part of any number measured through it."
        ) from exc
    return SimpleNamespace(
        cutlass=cutlass,
        cute=cute,
        cute_ext=cute_ext,
        sm100_utils=sm100_utils,
        from_dlpack=from_dlpack,
    )


@lru_cache(maxsize=1)
def _kernel_class():  # noqa: ANN202 - the class does not exist until the DSL is imported
    """Build the DSL kernel class. Cached: the class object is what the DSL keys its JIT cache on."""
    dsl = require_cute_dsl()
    cutlass, cute = dsl.cutlass, dsl.cute
    cute_ext, sm100_utils = dsl.cute_ext, dsl.sm100_utils

    class B_R5DenseGemm:
        """D = A @ B on sm_100a: TMA loads -> tcgen05.mma into TMEM -> TMEM->RMEM->SMEM->GMEM.

        Structure follows ``dense_gemm.py`` at 59e3a333 step for step, and every line reference is
        to that file. Two deliberate divergences, both so the rung stays a measurement rather than
        a library: the mainloop depth is a static Python constant (:func:`mainloop_stages`) instead
        of a device query, and there is no ``epilogue_op`` — an elementwise op folded into the
        epilogue would change the FLOP count the metric divides by.
        """

        def __init__(self, cfg: TileConfig, *, mainloop_stage: int):
            self.cfg = cfg
            self.mn_tiler = cfg.mma_tile_mn
            self.ab_dtype = cutlass.BFloat16
            self.acc_dtype = cutlass.Float32
            self.tmem_output_dtype = cutlass.Float32
            self.use_2cta_instrs = cfg.use_2cta_instrs
            self.cluster_shape = (*cfg.cluster_shape_mn, 1)
            self.mainloop_stage = mainloop_stage
            self.acc_stage = cfg.acc_stage
            self.tma_store_stage = TMA_STORE_STAGE

        @cute.experimental.jit
        def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mD: cute.Tensor):
            """Host launcher, ``dense_gemm.py:236-331``. One CTA per output tile; no persistence."""
            cta_tile_m = self.mn_tiler[0] // 2 if self.use_2cta_instrs else self.mn_tiler[0]
            grid = cute.round_up(  # :280-287
                (
                    cute.ceil_div(mD.layout.shape[0], cta_tile_m),
                    cute.ceil_div(mD.layout.shape[1], self.mn_tiler[1]),
                    mD.layout.shape[2],
                ),
                self.cluster_shape,
            )
            # The launch opts into the FULL sm_100 shared-memory capacity (:331). Everything above
            # 48 KB needs that opt-in and this kernel is far above it — see the smem budget test.
            self.kernel(mA, mB, mD).launch(  # :327-331
                grid=grid,
                block=(THREADS_PER_CTA, 1, 1),
                cluster=self.cluster_shape,
                smem=cute.Int64(cutlass.memory.get_smem_capacity_in_bytes("sm_100")),
            )

        @cute.experimental.kernel
        def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mD: cute.Tensor):
            """Device side. Setup transcribed from ``dense_gemm.py:334-855``; then the hole."""
            # --- the MMA atom and the tile it works on, :383-423 -----------------------------
            if cutlass.const_expr(self.use_2cta_instrs):
                cta_group = cute.nvgpu.tcgen05.CtaGroup.TWO
            else:
                cta_group = cute.nvgpu.tcgen05.CtaGroup.ONE
            tiled_mma = sm100_utils.make_trivial_tiled_mma(  # :387-395
                self.ab_dtype,
                self.ab_dtype,
                cutlass.tensor_utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
                cutlass.tensor_utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
                self.acc_dtype,
                cta_group,
                self.mn_tiler,
            )
            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])  # 16 for bf16, bh:1198
            mnk_tiler = (self.mn_tiler[0], self.mn_tiler[1], mma_inst_shape_k * MMA_INST_TILE_K)

            d_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(mD)
            d_dtype = mD.element_type

            num_mma_ctas = cute.size(tiled_mma.thr_id.shape)  # :428-435
            cta_tile_shape_mnk = cute.shape_div(mnk_tiler, (num_mma_ctas, 1, 1))
            a_tiler_mk = (cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])
            b_tiler_nk = (cta_tile_shape_mnk[1] // num_mma_ctas, cta_tile_shape_mnk[2])
            c_tiler_mn = (cta_tile_shape_mnk[0], cta_tile_shape_mnk[1])

            gA = cute.zipped_divide(mA, a_tiler_mk)  # :462-464
            gB = cute.zipped_divide(mB, b_tiler_nk)
            gD = cute.zipped_divide(mD, c_tiler_mn)

            # --- who am I, and which CTA of the pair, :488-505 -------------------------------
            cta_m, cta_n, cta_l = cute.arch.block_idx()
            tid_x, _, _ = cute.arch.thread_idx()
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape),
                cute.core._pack_shape((cute.size(tiled_mma.thr_id.shape),)),
            )
            cluster_layout_v_size = cute.size(cluster_layout_vmnk.shape[0])
            mma_coord_v = cta_m % cluster_layout_v_size
            # Under cta_group::2 only the leader CTA issues the MMA; the peer stages operands and
            # drains its half of the accumulator (:504, :935).
            is_leader_cta = mma_coord_v == 0
            thr_mma = tiled_mma.get_slice(mma_coord_v)

            gA_mkl = cute.local_tile(  # :509-514
                mA, cute.slice_(mnk_tiler, (None, 0, None)), (None, None, None)
            )
            gB_nkl = cute.local_tile(
                mB, cute.slice_(mnk_tiler, (0, None, None)), (None, None, None)
            )
            tCgA = thr_mma.partition_A(gA_mkl)
            tCgB = thr_mma.partition_B(gB_nkl)
            mma_tile_coord_m = cta_m // cluster_layout_v_size
            tAgA_slice = tCgA[(None, None, None, mma_tile_coord_m, None, cta_l)]
            tBgB_slice = tCgB[(None, None, None, cta_n, None, cta_l)]
            gD_tile = gD[(None, None), (cta_m, cta_n, cta_l)]

            # --- SMEM layouts, :539-550. The swizzle is chosen inside these helpers from the
            # K-tile's width in bits (bh:662-722): 64 bf16 elements = 1024 bits -> K_SW128.
            a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, mnk_tiler, self.ab_dtype, self.mainloop_stage
            )
            b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, mnk_tiler, self.ab_dtype, self.mainloop_stage
            )

            # --- the epilogue sub-tile and its staging buffer, :572-589 ----------------------
            epi_tile = sm100_utils.compute_epilogue_tile_shape(
                cta_tile_shape_mnk, self.use_2cta_instrs, d_layout, d_dtype
            )
            sc_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                d_dtype, d_layout, epi_tile, self.tma_store_stage
            )

            # --- the accumulator: TMEM, not registers, :606-648 ------------------------------
            tmem_layout = cute_ext.make_tmem_layout_acc(tiled_mma, mnk_tiler, self.acc_stage)
            buffer_a = cute_ext.allocate(
                self.ab_dtype, cutlass.AddressSpace.smem, a_smem_layout_staged, alignment=1024
            )
            buffer_b = cute_ext.allocate(
                self.ab_dtype, cutlass.AddressSpace.smem, b_smem_layout_staged, alignment=1024
            )
            # is2cta is not cosmetic: it selects the 2-CTA TMEM allocation, whose deallocation is
            # mbarrier-synchronised across the pair (cutlass/memory/tmem.py:283-301, :435-455).
            buffer_acc = cute_ext.allocate(
                self.acc_dtype,
                cutlass.AddressSpace.tmem,
                tmem_layout,
                alignment=16,
                is2cta=self.use_2cta_instrs,
            )
            buffer_c = cute_ext.allocate(
                d_dtype, cutlass.AddressSpace.smem, sc_smem_layout_staged, alignment=1024
            )

            # --- TMEM -> RMEM copy, :672-746. This hop does not exist on Hopper. --------------
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                cta_tile_shape_mnk,
                d_layout,
                self.tmem_output_dtype,
                self.acc_dtype,
                epi_tile,
                self.use_2cta_instrs,
            )
            accumulators = cute.zipped_divide(buffer_acc, ((epi_tile), 1))  # :700-701
            acc_epi_div = accumulators[((None, None), 0), 0]
            tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_epi_div)  # :708
            gC_mnl_epi = cute.flat_divide(gD_tile, epi_tile)  # :727-730
            acc_d_rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gC_mnl_epi, tid_x)
            buffer_r_acc = cute_ext.allocate(
                self.acc_dtype, cutlass.AddressSpace.rmem, acc_d_rmem_layout, alignment=32
            )
            buffer_r_d = cute_ext.allocate(
                d_dtype, cutlass.AddressSpace.rmem, acc_d_rmem_layout, alignment=32
            )

            # --- pipelines, :755-848. There is no explicit TMA atom to build in this DSL: the
            # descriptor is constructed inside cute_ext.tma_load itself
            # (cute/experimental/memory.py:176-224), so what is built here is the pipeline that
            # sequences those loads, plus the multicast projections they are issued with.
            tma_mcast_proj_a = 2  # A is shared across N, :762-764
            tma_mcast_proj_b = 1  # B is shared across M
            if cutlass.const_expr(self.use_2cta_instrs):
                mma_operation_type = cute_ext.OperationTypeEnum.SM100_MMA_2SM_SS
                tma_operation_type = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM
            else:
                mma_operation_type = cute_ext.OperationTypeEnum.SM100_MMA_1SM_SS
                tma_operation_type = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
            mainloop_pipe = cute_ext.TMAToUMMAPipeline.create(  # :786-791
                num_stages=self.mainloop_stage,
                mma_operation_type=mma_operation_type,
                tma_operation_type=tma_operation_type,
                cluster_layout_vmnk=cluster_layout_vmnk,
            )
            # 256 consumers under cta_group::2: both CTAs' epilogue warpgroups release the stage.
            acc_pipe = cute_ext.UMMAtoAsyncPipeline.create(  # :801-808
                num_stages=self.acc_stage,
                mma_operation_type=mma_operation_type,
                consumer=cute_ext.OperationTypeEnum.SM100_COPY_T2R,
                consumer_arv_count=256 if self.use_2cta_instrs else 128,
                cluster_layout_vmnk=cluster_layout_vmnk,
            )

            # --- warp specialisation, :826-848 -----------------------------------------------
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tma_store_pipe = cute_ext.TMAStorePipeline(  # :843-848
                stages=self.tma_store_stage,
                arv_count=128,
                barrier_id=1,
                tma_warp_id=WARP_ROLE_TMA_STORE,
            )

            is_tma_thr = warp_idx == WARP_ROLE_TMA_LOAD
            is_mma_thr = warp_idx == WARP_ROLE_MMA
            is_epi_thr = warp_idx < len(WARP_ROLE_EPILOGUE)
            k_tile_count = cute.size(gA, mode=[1, 1])  # :855

            # THE RUNG STARTS HERE. Everything above is scaffolding transcribed from the reference;
            # what follows is the block this ladder never delegates. It stays INLINE in `kernel`
            # rather than moving to a helper method because the DSL's AST preprocessor rewrites
            # only the DECORATED function (docs/pythonDSL/cute_dsl_general/dsl_code_generation.rst
            # :114); inside an undecorated callee, `if is_tma_thr:` would be a native Python branch
            # on an IR value, which is an error (dsl_control_flow.rst:20). Every cute_ext example
            # keeps it inline for that reason.
            #
            # Read map.md items 9-11 first — producer :866-924, MMA :935-1026, epilogue :1043-1102:
            #
            #   warp 5 (is_tma_thr): per k-tile, producer_acquire_and_get_stage -> get_mbarrier ->
            #     two cute_ext.tma_load (A with multicast_mode=tma_mcast_proj_a, B with _b, both
            #     vmnk_layout=cluster_layout_vmnk, tma_operation_type=tma_operation_type) ->
            #     producer_commit_and_advance.
            #   warp 4, leader CTA only (is_mma_thr and is_leader_cta): reserve an acc_pipe stage,
            #     slice buffer_acc, mma_atom = cute.make_mma_atom(tiled_mma.op), ACCUMULATE False,
            #     then per k-tile consumer_wait, MMA_INST_TILE_K x cute_ext.dot(...), and flip
            #     ACCUMULATE True after the FIRST dot (:1018) — after the first INSTRUCTION, not
            #     the first k-tile; flipped a loop too late it silently adds one k-tile's product
            #     to whatever TMEM held before.
            #   warps 0-3 (is_epi_thr): consumer_wait on acc_pipe, flat_divide the accumulator by
            #     epi_tile, then per sub-tile TMEM->RMEM (partition_and_copy through
            #     tiled_copy_t2r.get_slice(tid_x) into buffer_r_acc), cast in registers into
            #     buffer_r_d, RMEM->SMEM into buffer_c, tma_store into gC_mnl_epi from warp
            #     WARP_ROLE_TMA_STORE only, then tma_store_pipe.tail() and
            #     acc_pipe.consumer_release_and_advance(). Under cta_group::2 BOTH CTAs' warps
            #     release that stage, which is what consumer_arv_count=256 above encodes.
            #
            # No epilogue_op, on purpose: an elementwise op folded into the epilogue would change
            # the FLOP count the metric divides by.
            #
            # The tuple below is the complete set of names the body has in scope. It exists only so
            # the scaffolding above does not read as dead code while the hole is open, and it is
            # the first line to delete when the body is written.
            _hole_scope = (
                is_tma_thr,
                is_mma_thr,
                is_epi_thr,
                is_leader_cta,
                k_tile_count,
                tid_x,
                tAgA_slice,
                tBgB_slice,
                gD_tile,
                gB,
                gC_mnl_epi,
                tiled_mma,
                d_dtype,
                buffer_a,
                buffer_b,
                buffer_c,
                buffer_acc,
                buffer_r_acc,
                buffer_r_d,
                tiled_copy_t2r,
                epi_tile,
                mainloop_pipe,
                acc_pipe,
                tma_store_pipe,
                cluster_layout_vmnk,
                tma_mcast_proj_a,
                tma_mcast_proj_b,
                tma_operation_type,
            )
            # HUY: the tcgen05 mainloop (warp 5 tma_load into a mainloop_pipe stage; warp 4, leader CTA only, cute_ext.dot into the TMEM accumulator) and the TMEM->RMEM->SMEM->GMEM epilogue on warps 0-3 — spec: experiments/K1/B-R5/spec.md — fill before B-R5
            raise NotImplementedError("HUY: B-R5 mainloop + epilogue unwritten")

    return B_R5DenseGemm


@lru_cache(maxsize=2)
def _kernel_instance(cta_group: int):  # noqa: ANN202 - opaque DSL object
    cfg = tile_config(cta_group)
    return _kernel_class()(cfg, mainloop_stage=mainloop_stages(cta_group))


#: One compiled handle per cta_group. Not an lru_cache because the key must not include the CuTe
#: tensors (they are unhashable and would pin device memory).
_COMPILED: dict[int, object] = {}


def _compiled_kernel(cta_group: int, dsl: SimpleNamespace, mA, mB, mD):  # noqa: ANN001,ANN202
    """``cute_ext.compile`` once, then reuse — ``dense_gemm.py:1290-1299``, ``:1370-1372``.

    A JIT compile inside a timing window is a measurement of the compiler. The tensors are marked
    layout-dynamic in :func:`_make_cute_tensors`, so one compiled kernel serves every shape this
    rung measures and the cache key is the cta_group alone.
    """
    if cta_group not in _COMPILED:
        compiled = dsl.cute_ext.compile(_kernel_instance(cta_group), mA, mB, mD)
        compiled.engine.initialize()
        _COMPILED[cta_group] = compiled
    return _COMPILED[cta_group]


def _make_cute_tensors(dsl: SimpleNamespace, a: Tensor, b: Tensor, out: Tensor):  # noqa: ANN202
    """Wrap the torch tensors as the (M,K,L) / (N,K,L) / (M,N,L) CuTe tensors the kernel expects.

    ``unsqueeze(0).permute(1, 2, 0)`` builds the (rows, cols, L) shape ``cutlass_torch.matrix``
    produces (``dense_gemm.py:1155-1158``) while leaving the two real modes' strides untouched —
    ``unsqueeze(-1)`` would instead give the L mode stride 1 and put it ahead of the data modes,
    which is not what a batched descriptor expects even at L=1. ``leading_dim`` names the stride-1
    mode (K for A, N for the (N,K) view of B, N for D), which is what the DSL turns into each
    operand's major mode via ``LayoutEnum.from_tensor(...).mma_major_mode()``.
    """
    from_dlpack = dsl.from_dlpack
    mA = from_dlpack(a.unsqueeze(0).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    mB = from_dlpack(b.t().unsqueeze(0).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=0
    )
    mD = from_dlpack(out.unsqueeze(0).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    return mA, mB, mD


def b_r5_gemm(a: Tensor, b: Tensor, *, cta_group: int = 1) -> Tensor:
    """``C = A @ B`` for bf16 ``a`` [M,K] and ``b`` [K,N]; fp32-accumulated, **fp32 output**.

    fp32 out, not bf16, for H-R1's reason: the accumulator is fp32 and rounding it on the way out
    would fold the kernel's error together with the output cast's, leaving the correctness gate
    unable to tell a TMEM addressing bug from a rounding difference. It is not free here — an fp32
    D doubles the epilogue's SMEM staging buffer and quarters the TMA store's element granularity
    against a bf16 D — and paying that at a correctness-first rung is the intended trade.

    ``cta_group`` is 1 (one CTA per MMA) or 2 (``cta_group::2``, the CTA pair). Both run the same
    128x256x64 tile; 2 halves the per-CTA M tile and the TMEM column cost. The plan row measures 1
    first and then 2, as two rows against the same floor.

    ``K`` must be a multiple of ``TILE_K`` (no K-remainder path at this rung), and the shape must
    be TMA-addressable — see :func:`tma_alignment_violations`.

    Raises ``RuntimeError`` off sm_100, ``ImportError`` without the DSL, ``ValueError`` for a shape
    TMA cannot address, and ``NotImplementedError`` while the kernel body is an open hole.

    The arch gate fires FIRST, before the DSL import: on a wrong-arch box the DSL imports happily
    and fails much later inside the JIT, and "this is not a B200" is the message that saves the hour.
    """
    require_arch(ARCH, fn_name="b_r5_gemm")
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError(
            f"b_r5_gemm expects bfloat16 operands (tcgen05 MmaF16BF16Op), got {a.dtype} and "
            f"{b.dtype}; the K1 floor is cuBLASLt bf16, so another dtype measures against no floor"
        )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"b_r5_gemm expects 2-D operands, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"b_r5_gemm inner dimensions disagree: {a.shape[1]} vs {b.shape[0]}")
    m, k = a.shape
    n = b.shape[1]
    cfg = tile_config(cta_group)
    if k % cfg.tile_k:
        raise ValueError(
            f"b_r5_gemm requires K % {cfg.tile_k} == 0 at this rung (no K-remainder path); got K={k}"
        )
    violations = tma_alignment_violations(m, n, k, cfg=cfg)
    if violations:
        raise ValueError(
            f"b_r5_gemm cannot address {m}x{n}x{k} with TMA: {'; '.join(violations)}. That is a "
            "property of the shape, not of the kernel — measure a shape the descriptor can encode "
            "or pad outside the timed region; a .contiguous() here would be timed as part of the GEMM."
        )

    dsl = require_cute_dsl()
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    mA, mB, mD = _make_cute_tensors(dsl, a, b, out)
    _compiled_kernel(cta_group, dsl, mA, mB, mD)(mA, mB, mD)
    return out


def b_r5_gemm_2cta(a: Tensor, b: Tensor) -> Tensor:
    """:func:`b_r5_gemm` at ``cta_group::2`` — the plan row's second half, same tile, same floor."""
    return b_r5_gemm(a, b, cta_group=2)


def reference_gemm(a: Tensor, b: Tensor) -> Tensor:
    """The oracle: the same bf16 inputs, multiplied in fp32.

    Identical in form to H-R1's, and for the same reason: it up-casts the operands the kernel
    actually receives, so the only difference between it and the kernel is accumulation order and
    the tensor core's internal rounding. An fp32-input reference would fold the operands' own bf16
    quantization error into the tolerance and hide a real TMEM-addressing bug under it.
    """
    return torch.matmul(a.float(), b.float())
