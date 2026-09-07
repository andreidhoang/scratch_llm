"""Hopper/Blackwell hardware contracts, as executable arithmetic — checkable without a GPU.

Most of what goes wrong in a warpgroup-MMA or TMA kernel is not a logic bug. It is an arithmetic
one: a descriptor field packed at the wrong bit, a swizzle that does not agree with the layout the
descriptor declares, a stage count whose shared memory does not fit, a tile that leaves half the
SMs idle on the last wave. Every one of those is decidable on a laptop, in milliseconds, from
numbers published in the PTX ISA and the CUDA programming guide — and every one of them, if left
to be discovered on a rented H100, costs an hour of silicon and produces a wrong number rather
than an error.

So this module is the executable form of those rules. It computes what the hardware requires; the
tests next to it assert the kernels agree; and `infra/drydock.sh` closes the loop by proving the
generated code matches (registers, spills, shared memory) before anything is rented.

Nothing here imports torch or CUDA. It is pure integer arithmetic over documented constants, which
is exactly why it can be trusted as a reference: there is no device for it to be wrong about.

Sources, all first-party and all quoted at the point of use below:
  * PTX ISA 8.x, "Asynchronous Warpgroup Level Matrix Multiply-Accumulate" — the 64-bit shared
    memory matrix descriptor bit layout, and `setmaxnreg` legality.
  * CUDA C Programming Guide, "Compute Capabilities" table — per-arch shared memory and register
    ceilings.
  * CUDA Driver API, `cuTensorMapEncodeTiled` — TMA tensor-map constraints.
  * CUTLASS `cute/atom/mma_traits_sm90_gmma.hpp` (GmmaDescriptor) and `oss/fast.cu` — the two
    independent implementations these values were cross-checked against.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# =============================================================================================
# Per-arch ceilings
# =============================================================================================


@dataclass(frozen=True)
class ArchLimits:
    """The ceilings a kernel for one architecture must stay under.

    ``smem_per_cta`` is the number the pipeline designer actually budgets against, and it is NOT
    ``smem_per_sm``: Hopper has 228 KB of shared memory per SM but caps a single CTA at 227 KB, and
    anything above 48 KB additionally requires an explicit
    ``cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`` opt-in at
    launch. ptxas will happily emit a kernel that asks for more and the failure appears only as a
    launch error at runtime — which is why :func:`check_smem_budget` exists.
    """

    name: str
    cc: tuple[int, int]
    smem_per_cta: int  # bytes, max a single CTA may opt into
    smem_per_sm: int  # bytes, total per SM (bounds CTAs/SM together with the above)
    regs_per_sm: int  # 32-bit registers per SM
    regs_per_thread_max: int  # architectural cap on a single thread's register count
    max_threads_per_cta: int
    smem_opt_in_threshold: int = 49152  # above this, cudaFuncSetAttribute is mandatory


# Compute Capabilities table, CUDA C Programming Guide. sm_120 (consumer Blackwell: RTX 5090,
# RTX PRO Blackwell) is a different machine from sm_100 (B200) despite the shared "Blackwell"
# name — 100 KB of shared memory per SM against 228, and no tcgen05. Routing a rung to "the
# biggest card" instead of to its arch is how a kernel silently stops fitting.
ARCH: dict[str, ArchLimits] = {
    "sm_90a": ArchLimits("sm_90a", (9, 0), 227 * 1024, 228 * 1024, 65536, 255, 1024),
    "sm_100a": ArchLimits("sm_100a", (10, 0), 227 * 1024, 228 * 1024, 65536, 255, 1024),
    "sm_120a": ArchLimits("sm_120a", (12, 0), 99 * 1024, 100 * 1024, 65536, 255, 1024),
}


def arch_for(cc: tuple[int, int]) -> ArchLimits:
    """The accelerated-ISA limits for a compute capability, e.g. ``(9, 0)`` -> ``sm_90a``."""
    for a in ARCH.values():
        if a.cc == cc:
            return a
    raise KeyError(f"no arch limits for compute capability {cc}; add it to ARCH with a citation")


# =============================================================================================
# The 64-bit shared-memory matrix descriptor (wgmma SS/RS operand form)
# =============================================================================================


class SwizzleMode(IntEnum):
    """``layout_type`` in the descriptor's top two bits. The value names a swizzle ATOM WIDTH."""

    NONE = 0
    B128 = 1
    B64 = 2
    B32 = 3

    @property
    def atom_bytes(self) -> int:
        """The width in bytes of the swizzle atom this mode describes (0 for NONE)."""
        return {
            SwizzleMode.NONE: 0,
            SwizzleMode.B128: 128,
            SwizzleMode.B64: 64,
            SwizzleMode.B32: 32,
        }[self]


#: The descriptor's three offset fields are each 14 bits holding a 16-byte-granular value.
#: PTX ISA and CUTLASS both express the packing as ``(x & 0x3FFFF) >> 4``: mask to 18 bits, drop
#: the low 4. Dropping the low 4 bits is not a rounding convenience — it is the hardware asserting
#: that every address and offset it is handed is 16-byte aligned.
_FIELD_MASK = 0x3FFFF
_FIELD_SHIFT = 4


def descriptor_encode(x: int) -> int:
    """Pack one 16-byte-granular value into its 14-bit descriptor field."""
    return (x & _FIELD_MASK) >> _FIELD_SHIFT


def wgmma_smem_descriptor(
    smem_byte_addr: int,
    *,
    leading_byte_offset: int,
    stride_byte_offset: int,
    base_offset: int = 0,
    swizzle: SwizzleMode = SwizzleMode.B128,
) -> int:
    """Build the 64-bit descriptor a ``wgmma.mma_async`` SS-form operand takes.

    Bit layout, PTX ISA "Matrix Descriptor Format"::

        [ 0, 14)  start address        (byte addr in the CTA shared window, >> 4)
        [14, 16)  --
        [16, 30)  leading byte offset  (>> 4)
        [30, 32)  --
        [32, 46)  stride byte offset   (>> 4)
        [46, 49)  --
        [49, 52)  matrix base offset   (the intra-atom swizzle phase, 0..7)
        [52, 62)  --
        [62, 64)  layout type          (SwizzleMode)

    ``smem_byte_addr`` is the address in the CTA's shared-memory window — what
    ``__cvta_generic_to_shared`` returns in CUDA, not a host pointer.

    The two offsets are *not* interchangeable and getting them the wrong way round is the single
    most common descriptor bug: the leading offset walks between the two 8x8 core matrices along
    the contiguous direction, the stride offset walks between consecutive core-matrix rows. For
    the canonical 128B-swizzled K-major bf16 tile they are 16 and 1024 bytes respectively (a value
    you can see identically in ``oss/fast.cu/h100/matmul/matmul_12.cuh`` and in CUTLASS's
    ``GmmaDescriptor``).
    """
    if not 0 <= base_offset <= 7:
        raise ValueError(f"base_offset must be a 3-bit swizzle phase 0..7, got {base_offset}")
    for label, v in (
        ("smem_byte_addr", smem_byte_addr),
        ("leading_byte_offset", leading_byte_offset),
        ("stride_byte_offset", stride_byte_offset),
    ):
        if v % 16:
            raise ValueError(
                f"{label}={v} is not 16-byte aligned; the descriptor's >>4 packing would silently "
                f"discard the low bits and the MMA would read the wrong core matrix"
            )
    desc = descriptor_encode(smem_byte_addr)
    desc |= descriptor_encode(leading_byte_offset) << 16
    desc |= descriptor_encode(stride_byte_offset) << 32
    desc |= (base_offset & 0x7) << 49
    desc |= int(swizzle) << 62
    return desc


def descriptor_fields(desc: int) -> dict[str, int]:
    """Unpack a descriptor — the inverse of :func:`wgmma_smem_descriptor`, for tests and diffs.

    Values come back in their encoded (16-byte-granular) form, which is what you compare when
    byte-diffing against ``cute::make_gmma_desc`` output captured on a box.
    """
    return {
        "start_address": desc & 0x3FFF,
        "leading_byte_offset": (desc >> 16) & 0x3FFF,
        "stride_byte_offset": (desc >> 32) & 0x3FFF,
        "base_offset": (desc >> 49) & 0x7,
        "layout_type": (desc >> 62) & 0x3,
    }


# =============================================================================================
# Swizzle
# =============================================================================================

#: Shared memory is 32 banks of 4 bytes. Two addresses in the same bank, in the same instruction,
#: from different threads, and not the same word, serialise. That is the entire cost model.
SMEM_BANKS = 32
SMEM_BANK_BYTES = 4


def swizzle_128b(byte_offset: int) -> int:
    """CuTe ``Swizzle<3,4,3>`` — the 128-byte GMMA swizzle atom.

    XOR bits [7,10) of the offset into bits [4,7)::

        swz(off) = off ^ (((off >> 7) & 0x7) << 4)

    The store into shared memory must apply this and the descriptor must declare
    :attr:`SwizzleMode.B128`; the two are one contract, and satisfying only one of them produces a
    kernel that runs, is fast, and is wrong.
    """
    return byte_offset ^ (((byte_offset >> 7) & 0x7) << 4)


def bank_of(byte_offset: int) -> int:
    """Which of the 32 shared-memory banks a byte offset lands in."""
    return (byte_offset // SMEM_BANK_BYTES) % SMEM_BANKS


def is_bijection(fn, span_bytes: int, *, granularity: int = 16) -> bool:
    """True iff ``fn`` permutes a ``span_bytes`` region — i.e. it relocates, never aliases.

    A swizzle that is not a bijection has two logical elements sharing one physical address: the
    kernel then silently reads one of them twice. This is checkable exhaustively over one atom.
    """
    offs = range(0, span_bytes, granularity)
    out = [fn(o) for o in offs]
    return len(set(out)) == len(out) and set(out) == set(offs)


def bank_conflict_degree(byte_offsets: list[int]) -> int:
    """Worst-case ways one bank is hit across ``byte_offsets`` (1 == conflict-free).

    Pass the offsets a single instruction issues across a warp's 32 lanes. The multi-cast case
    (all lanes reading the identical address) is a broadcast, not a conflict, and is excluded.
    """
    if not byte_offsets:
        return 1
    per_bank: dict[int, set[int]] = {}
    for off in byte_offsets:
        per_bank.setdefault(bank_of(off), set()).add(off // SMEM_BANK_BYTES)
    return max(len(words) for words in per_bank.values())


# =============================================================================================
# TMA (cuTensorMapEncodeTiled)
# =============================================================================================


def check_tma_tensor_map(
    *,
    rank: int,
    elem_bytes: int,
    global_dims: list[int],
    global_strides_bytes: list[int],
    box_dims: list[int],
    swizzle: SwizzleMode,
    global_address: int = 0,
) -> list[str]:
    """Every ``cuTensorMapEncodeTiled`` constraint, as a list of violations (empty == legal).

    The driver returns ``CUDA_ERROR_INVALID_VALUE`` for any of these with no indication of which
    one, and the call is on the host at startup — so a violation shows up as a kernel that never
    launches, hours after the tile shape that caused it was chosen. Checking them here turns that
    into a named assertion at CPU test time.

    ``global_strides_bytes`` holds the strides for dimensions 1..rank-1 (the innermost dimension is
    implicitly contiguous and has no entry), matching the driver's own argument convention.
    """
    errs: list[str] = []
    if not 1 <= rank <= 5:
        errs.append(f"rank {rank} outside the supported 1..5")
    if len(box_dims) != rank:
        errs.append(f"box_dims has {len(box_dims)} entries for rank {rank}")
    if global_address % 16:
        errs.append(f"globalAddress {global_address} is not 16-byte aligned")
    for i, b in enumerate(box_dims):
        if not 1 <= b <= 256:
            errs.append(f"boxDim[{i}]={b} outside 1..256")
    for i, s in enumerate(global_strides_bytes):
        if s % 16:
            errs.append(f"globalStrides[{i}]={s} is not a multiple of 16 bytes")
    if swizzle is not SwizzleMode.NONE and box_dims:
        inner_bytes = box_dims[0] * elem_bytes
        if inner_bytes != swizzle.atom_bytes:
            errs.append(
                f"innermost box is {inner_bytes} B but {swizzle.name} swizzle requires exactly "
                f"{swizzle.atom_bytes} B — split the dimension into "
                f"{swizzle.atom_bytes // elem_bytes}-element chunks and add a rank, the way "
                f"create_tensor_map does in oss/fast.cu/h100/matmul/matmul_12.cuh"
            )
    if len(global_dims) != rank:
        errs.append(f"global_dims has {len(global_dims)} entries for rank {rank}")
    return errs


# =============================================================================================
# Occupancy: shared memory, registers, waves
# =============================================================================================


def check_smem_budget(
    *, bytes_per_stage: int, stages: int, arch: ArchLimits, extra_bytes: int = 0
) -> list[str]:
    """Violations of the shared-memory budget for a multistage pipeline (empty == fits).

    ``extra_bytes`` covers everything that is not a pipeline stage: mbarriers, the epilogue's
    staging buffer, a tile scheduler's scratch.
    """
    total = bytes_per_stage * stages + extra_bytes
    errs: list[str] = []
    if total > arch.smem_per_cta:
        errs.append(
            f"{stages} stages x {bytes_per_stage} B + {extra_bytes} B = {total} B exceeds "
            f"{arch.name}'s {arch.smem_per_cta} B per-CTA cap — drop to "
            f"{max(0, (arch.smem_per_cta - extra_bytes) // bytes_per_stage)} stages or shrink the tile"
        )
    elif total > arch.smem_opt_in_threshold:
        # Not an error: legal, but only if the launch opts in. Reported so a kernel author cannot
        # forget the cudaFuncSetAttribute call and meet it as a launch failure instead.
        errs.append(
            f"OPT-IN REQUIRED: {total} B > {arch.smem_opt_in_threshold} B — the launch must call "
            f"cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, {total})"
        )
    return errs


def max_stages(*, bytes_per_stage: int, arch: ArchLimits, extra_bytes: int = 0) -> int:
    """How many pipeline stages actually fit. The answer the tile shape has to be chosen against."""
    if bytes_per_stage <= 0:
        raise ValueError("bytes_per_stage must be positive")
    return max(0, (arch.smem_per_cta - extra_bytes) // bytes_per_stage)


def check_register_budget(
    *, regs_per_thread: int, threads_per_cta: int, arch: ArchLimits
) -> list[str]:
    """Violations of the register budget for one CTA (empty == fits)."""
    errs: list[str] = []
    if regs_per_thread > arch.regs_per_thread_max:
        errs.append(
            f"{regs_per_thread} registers/thread exceeds the architectural max {arch.regs_per_thread_max}"
        )
    if threads_per_cta > arch.max_threads_per_cta:
        errs.append(f"{threads_per_cta} threads/CTA exceeds {arch.max_threads_per_cta}")
    total = regs_per_thread * threads_per_cta
    if total > arch.regs_per_sm:
        errs.append(
            f"{regs_per_thread} x {threads_per_cta} = {total} registers exceeds the {arch.regs_per_sm} "
            f"per SM — this CTA cannot be resident even one at a time"
        )
    return errs


def check_setmaxnreg(n: int) -> list[str]:
    """``setmaxnreg`` legality (empty == legal).

    PTX ISA: the immediate must be in [24, 256] and a multiple of 8. Warp specialisation lives or
    dies on this instruction — the producer warpgroup gives registers back (``.dec``) so the
    consumer warpgroups can take them (``.inc``) — and an illegal immediate is a compile error at
    best and a silently unspecialised kernel at worst.
    """
    errs: list[str] = []
    if not 24 <= n <= 256:
        errs.append(f"setmaxnreg {n} outside the legal range 24..256")
    if n % 8:
        errs.append(f"setmaxnreg {n} is not a multiple of 8")
    return errs


@dataclass(frozen=True)
class WaveQuantization:
    """How badly a tile shape fits the machine on its last wave."""

    ctas: int
    sm_count: int
    waves: float
    full_waves: int
    tail_ctas: int
    tail_utilization: float  # fraction of SMs busy during the final partial wave

    @property
    def efficiency(self) -> float:
        """Achieved fraction of peak from grid shape alone, ignoring every other effect.

        ``waves / ceil(waves)``. At 1.02 waves this is 0.51: half the machine idles for the whole
        second wave, and no amount of mainloop tuning recovers it. This is why the tile shape is
        chosen before the mainloop is optimised, not after.
        """
        import math

        return self.waves / math.ceil(self.waves) if self.waves else 0.0


def wave_quantization(
    *, m: int, n: int, tile_m: int, tile_n: int, sm_count: int, ctas_per_sm: int = 1
) -> WaveQuantization:
    """Grid-vs-machine fit for one problem shape and tile shape."""
    import math

    if min(m, n, tile_m, tile_n, sm_count, ctas_per_sm) <= 0:
        raise ValueError("all arguments must be positive")
    ctas = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    slots = sm_count * ctas_per_sm
    waves = ctas / slots
    full = ctas // slots
    tail = ctas % slots
    return WaveQuantization(
        ctas=ctas,
        sm_count=sm_count,
        waves=waves,
        full_waves=full,
        tail_ctas=tail,
        tail_utilization=(tail / slots) if tail else 1.0,
    )


def tile_covers(
    *, m: int, n: int, k: int, tile_m: int, tile_n: int, tile_k: int
) -> dict[str, bool]:
    """Which problem dimensions the tile divides exactly.

    A ``False`` is not a failure — it means that dimension needs predication (or the kernel must
    reject the shape). It is a failure only when the kernel does neither, which is precisely the
    bug the non-power-of-two test shape exists to catch.
    """
    return {"m": m % tile_m == 0, "n": n % tile_n == 0, "k": k % tile_k == 0}
