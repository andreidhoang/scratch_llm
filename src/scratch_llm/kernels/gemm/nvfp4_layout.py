"""NVFP4's scale-factor layout and its two number formats, as executable arithmetic — no GPU.

A block-scaled GEMM is a GEMM plus a second tensor, and that second tensor is where the rung is
won or lost. The scale factors are not stored row-major next to the data: CUTLASS lays them out in
a 512-byte indivisible block whose interior is permuted, so that any aligned 32 bits hold the four
K-blocks *of one row* — the TMEM word size on sm100, and the width of the single ``.b32`` scale
operand the sm120 ``mma.sync`` takes. Get the permutation wrong and nothing faults: the kernel runs
at full speed and scales the wrong 16 elements.

That permutation is pure integer arithmetic over published constants, so it is decidable here, on a
laptop, in microseconds — the same argument :mod:`scratch_llm.kernels.common.hopper_contracts` makes
for the wgmma descriptor. Nothing in this module imports torch or CUDA; the torch-side quantizer in
``kernels/gemm/cuda/b_r6.py`` calls into it, and the CPU tests assert the ``.cu``'s device-side
copy of :func:`sf_byte_offset` agrees with this one.

Sources, quoted at the point of use:
  * ``oss/cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp:51-55`` — ``Blk_MN=128``,
    ``Blk_SF=4``, and the K-major SF atom whose strides are the whole permutation.
  * ``…/sm100_blockscaled_layout.hpp:93,108`` — ``tile_to_shape(SfAtom, (M,K), Step<_2,_1>)``:
    how the atom repeats to cover the tensor.
  * ``oss/cutlass/include/cute/layout.hpp:1802-1812`` — what ``Step<_2,_1>`` means (repeat across
    the second mode first).
  * ``oss/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl:164``
    — sm120 reuses the identical ``Sm1xxBlockScaledConfig``, so this gmem layout is one layout for
    two architectures. Only the *shared-memory* SF layout differs (same file, :189-214).
  * ``oss/cutlass/include/cutlass/gemm/collective/builders/sm1xx_common.inl:470-475`` —
    ``SfVectorSize = 16`` for dense ``nv_float4_t``.
  * ``oss/cutlass/include/cutlass/exmy_base.h:937-939`` (E2M1) and :917-919 (UE4M3) — the two
    encodings' (bits, exponent, mantissa, sign) tuples, from which every value below is derived
    rather than tabulated by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# =============================================================================================
# The two number formats
# =============================================================================================


@dataclass(frozen=True)
class FpFormat:
    """A tiny floating-point encoding, and the finite values it can hold.

    Enough of one to derive a code table from, which is the point: NVFP4's error budget is set by
    where the gaps in these grids are, and a hand-typed table of magnitudes is a place for a typo
    to become a tolerance.

    ``bits`` is the STORAGE width and ``code_bits`` the width the encoding actually uses; for
    UE4M3 they differ (8 vs 7) because CUTLASS declares it ``IS_SIGNED=false`` with
    ``NumBits = 8`` (``exmy_base.h:917-919``), leaving the top storage bit unused. ``has_nan``
    follows ``NanInfEncoding``: ``CANONICAL_ONLY`` means the all-ones code is NaN and there is no
    Inf; ``NONE`` means every bit pattern is a finite number — which is exactly true of E2M1 and is
    the reason a NaN cannot survive NVFP4.
    """

    name: str
    bits: int
    exp_bits: int
    mant_bits: int
    signed: bool
    has_nan: bool

    def __post_init__(self) -> None:
        if self.bits < self.exp_bits + self.mant_bits + int(self.signed):
            raise ValueError(
                f"{self.name}: {self.bits} storage bits cannot hold e{self.exp_bits}m{self.mant_bits}"
            )

    @property
    def code_bits(self) -> int:
        """Bits the encoding uses. ``<= bits``; the difference is unused storage."""
        return self.exp_bits + self.mant_bits + int(self.signed)

    @property
    def bias(self) -> int:
        """``2^(E-1) - 1`` — the IEEE-style exponent bias every one of these encodings uses."""
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def has_subnormals(self) -> bool:
        """False only for a zero-mantissa format: E8M0 is all exponent, so it has no zero either."""
        return self.mant_bits > 0

    @property
    def nan_code(self) -> int | None:
        """The canonical NaN pattern (exponent and mantissa all ones), or ``None``.

        ``NAN_MASK = (EXPONENT_MASK << NUM_MANTISSA_BITS) | MANTISSA_MASK`` (``exmy_base.h:464``),
        which for the unsigned UE4M3 is 0x7F, not 0xFF — the sign bit is not part of the encoding.
        """
        return None if not self.has_nan else (1 << (self.exp_bits + self.mant_bits)) - 1

    def decode(self, code: int) -> float:
        """One bit pattern -> its value. Raises on the NaN code rather than returning ``nan``.

        A caller that wants NaN tolerance has to say so; silently handing back ``nan`` from a
        format lookup is how a poisoned scale factor becomes an all-NaN output tile that reads
        like a kernel bug.
        """
        if not 0 <= code < (1 << self.code_bits):
            raise ValueError(f"{self.name}: code {code} does not fit in {self.code_bits} bits")
        if code == self.nan_code:
            raise ValueError(f"{self.name}: code 0x{code:x} is the canonical NaN, not a value")
        sign = -1.0 if (self.signed and (code >> (self.exp_bits + self.mant_bits)) & 1) else 1.0
        mant = code & ((1 << self.mant_bits) - 1)
        exp = (code >> self.mant_bits) & ((1 << self.exp_bits) - 1)
        if not self.has_subnormals:  # E8M0: every code is a normal power of two, and 0 is not one
            return sign * math.ldexp(1.0, exp - self.bias)
        if exp == 0:  # subnormal: no implicit leading 1, exponent pinned at 1 - bias
            return sign * math.ldexp(mant, 1 - self.bias - self.mant_bits)
        return sign * math.ldexp((1 << self.mant_bits) | mant, exp - self.bias - self.mant_bits)

    def values(self) -> list[float]:
        """Every finite value, indexed by bit pattern; the NaN code holds ``inf`` as a sentinel.

        ``inf`` rather than ``nan`` so that a "smallest grid value >= v" search simply never
        selects it, without a special case at every call site.
        """
        return [
            float("inf") if c == self.nan_code else self.decode(c)
            for c in range(1 << self.code_bits)
        ]


#: The element format. E2M1: 1 sign, 2 exponent, 1 mantissa bit, **no NaN and no Inf**
#: (``NanInfEncoding::NONE``, ``exmy_base.h:937-939``). Its whole grid is
#: ``{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`` — eight magnitudes, and the gap between the last two is
#: 2.0, which is the single fact the rung's tolerance is built out of.
FP4_E2M1 = FpFormat("e2m1", bits=4, exp_bits=2, mant_bits=1, signed=True, has_nan=False)

#: NVFP4's scale format: unsigned E4M3 (``float_ue4m3_t``, ``float_subbyte.h:515``, encoding at
#: ``exmy_base.h:917-919``). Unsigned because a block scale is a magnitude; the sign bit is simply
#: not part of the encoding, so the byte holds e4m3 in bits [0,7) and bit 7 is unused. Max finite
#: 448, min positive 2^-9.
SF_UE4M3 = FpFormat("ue4m3", bits=8, exp_bits=4, mant_bits=3, signed=False, has_nan=True)

#: MXFP4's scale format, for contrast: a pure power of two (``float_ue8m0_t``,
#: ``float_subbyte.h:503``). It is here because the *layout* below is shared — ``mx_float4_t`` and
#: ``nv_float4_t`` differ only in this type and in ``SFVecSize`` — so a reader comparing the two
#: paths does not have to leave the module to see what changes. Having three mantissa bits rather
#: than none is exactly what NVFP4 buys over MXFP4, and what makes its scale rounding a real step.
SF_UE8M0 = FpFormat("ue8m0", bits=8, exp_bits=8, mant_bits=0, signed=False, has_nan=True)

#: Elements sharing one scale factor. 16 for dense NVFP4 (``sm1xx_common.inl:470-475``); 32 appears
#: only on the sparse schedules and on MXFP4's ``scale_vec::2X``.
SF_VEC_SIZE = 16


# =============================================================================================
# The scale-factor layout
# =============================================================================================

#: ``Blk_MN`` — rows (for SFA) or columns (for SFB) inside one indivisible block.
#: ``sm100_blockscaled_layout.hpp:51``.
BLK_MN = 128

#: ``Blk_SF`` — scale factors along K inside one block. 4, chosen upstream "to make consecutive
#: 32 bits of data have scale factors for only a single row (col)" (:134-135). That sentence is the
#: design: the sm120 ``mma.sync`` takes its scales as ONE ``.b32`` operand
#: (``cute/arch/mma_sm120.hpp:3194-3197``), so four consecutive bytes must be four K-blocks of one
#: row, and nothing else.
BLK_SF = 4

#: One atom covers ``BLK_MN`` rows x ``BLK_SF * SF_VEC_SIZE`` K-elements and occupies exactly
#: ``BLK_MN * BLK_SF`` bytes — 512, one byte per (row, K-block) pair, densely.
ATOM_ROWS = BLK_MN
ATOM_K = BLK_SF * SF_VEC_SIZE  # 64
ATOM_BYTES = BLK_MN * BLK_SF  # 512


def atom_offset(row_in_atom: int, sf_block_in_atom: int) -> int:
    """Byte offset inside one 512-byte atom. **The formula the whole rung rests on.**

    ``sm100_blockscaled_layout.hpp:54-55`` states the atom as a CuTe layout::

        Layout< Shape< Shape<_32,_4>, Shape<_16,_4>>,
               Stride<Stride<_16,_4>, Stride< _0,_1>> >

    Read left to right: the MN mode's index ``r`` splits as ``r = r0 + 32*r1`` (``r0 < 32``,
    ``r1 < 4``, so 128 rows) and contributes ``16*r0 + 4*r1``; the K mode's index splits as
    ``k = k0 + 16*k1`` and contributes ``0*k0 + 1*k1``. The stride-0 on ``k0`` is the block scaling
    itself, spelled as a layout: all 16 K-elements of a block read the identical byte.

        offset = 16*r0 + 4*r1 + k1,   max 16*31 + 4*3 + 3 = 511

    so the atom is dense in [0, 512) — see :func:`atom_is_bijection`. Two consequences are worth
    holding on to: consecutive bytes walk K (four of them, one row), and the +16 per ``r0``
    interleaves rows that are 32 apart, which is what lets a warp read 32 distinct rows' scales as
    32 distinct 4-byte words.
    """
    if not 0 <= row_in_atom < ATOM_ROWS:
        raise ValueError(f"row_in_atom {row_in_atom} outside [0, {ATOM_ROWS})")
    if not 0 <= sf_block_in_atom < BLK_SF:
        raise ValueError(f"sf_block_in_atom {sf_block_in_atom} outside [0, {BLK_SF})")
    r0, r1 = row_in_atom % 32, row_in_atom // 32
    return 16 * r0 + 4 * r1 + sf_block_in_atom


def atom_offset_inverse(offset: int) -> tuple[int, int]:
    """``offset -> (row_in_atom, sf_block_in_atom)``. The inverse of :func:`atom_offset`.

    Written as an inverse rather than a search because that is the direction a *debugger* needs:
    given the byte the kernel actually read, which row and which K-block did it think it was
    reading? Mixed radix, digits (16, 4, 1).
    """
    if not 0 <= offset < ATOM_BYTES:
        raise ValueError(f"offset {offset} outside one atom [0, {ATOM_BYTES})")
    r0, rest = divmod(offset, 16)
    r1, sf_block = divmod(rest, 4)
    return r0 + 32 * r1, sf_block


def atom_is_bijection() -> bool:
    """True iff :func:`atom_offset` permutes all 512 (row, K-block) pairs onto [0, 512).

    Exhaustive, 512 iterations, and it is the assertion that makes the byte-offset formula safe to
    build a kernel on: a layout that is not a bijection has two logical scale factors sharing one
    physical byte, and the kernel then scales one block with another block's factor — a wrong
    result that is not a crash, not a NaN, and not slow.
    """
    seen = {atom_offset(r, b) for r in range(ATOM_ROWS) for b in range(BLK_SF)}
    return len(seen) == ATOM_BYTES and seen == set(range(ATOM_BYTES))


def sf_atom_grid(rows: int, k: int) -> tuple[int, int]:
    """``(row_atoms, k_atoms)`` — how many atoms tile a ``rows x k`` operand.

    Both are ``ceil`` divisions: the atom is *indivisible*, so an operand with 130 rows still
    allocates two full 128-row atom rows, and that padding is real memory the TMA descriptor
    addresses. Sizing the allocation as ``rows*k/16`` instead is the allocation bug that surfaces
    as an illegal address only on the last tile of the last wave.
    """
    if rows <= 0 or k <= 0:
        raise ValueError(f"rows={rows}, k={k} must both be positive")
    if k % SF_VEC_SIZE:
        raise ValueError(
            f"k={k} is not a multiple of SF_VEC_SIZE={SF_VEC_SIZE}; a partial scale block has no "
            f"encoding — pad the operand's K instead"
        )
    return (rows + ATOM_ROWS - 1) // ATOM_ROWS, (k + ATOM_K - 1) // ATOM_K


def sf_tensor_bytes(rows: int, k: int) -> int:
    """Bytes the swizzled scale tensor occupies for a ``rows x k`` operand.

    ``512 * row_atoms * k_atoms``. Upstream sizes the same allocation as
    ``size(filter_zeros(layout_SFA))`` (``72a_blackwell_nvfp4_bf16_gemm.cu:353``); ``filter_zeros``
    drops the stride-0 K mode, leaving exactly ``BLK_MN * BLK_SF`` bytes per atom.
    """
    row_atoms, k_atoms = sf_atom_grid(rows, k)
    return ATOM_BYTES * row_atoms * k_atoms


def sf_byte_offset(row: int, k: int, *, rows: int, k_total: int) -> int:
    """``(row, k) -> byte offset`` in the swizzled scale tensor of a ``rows x k_total`` operand.

    The atom grid is **K-major**: ``tile_atom_to_shape_SFA`` tiles with ``Step<_2,_1>``
    (``sm100_blockscaled_layout.hpp:93``), and ``Step<_2,_1>`` means "repeat across the second mode
    first" (``cute/layout.hpp:1808-1811``), so K-atoms are contiguous and the M-atom stride is
    ``k_atoms`` atoms. Composed with :func:`atom_offset`::

        offset = 512 * (m_atom * k_atoms + k_atom) + 16*r0 + 4*r1 + k1

    ``rows`` is required rather than inferred because the *bound* on ``row`` depends on it, and
    silently accepting an out-of-range row would hand back a legal-looking offset that lands inside
    the next operand's scales.
    """
    if not 0 <= row < rows:
        raise ValueError(f"row {row} outside [0, {rows})")
    if not 0 <= k < k_total:
        raise ValueError(f"k {k} outside [0, {k_total})")
    _, k_atoms = sf_atom_grid(rows, k_total)
    m_atom, row_in_atom = divmod(row, ATOM_ROWS)
    k_atom, k_in_atom = divmod(k, ATOM_K)
    return ATOM_BYTES * (m_atom * k_atoms + k_atom) + atom_offset(
        row_in_atom, k_in_atom // SF_VEC_SIZE
    )


def sf_byte_offset_inverse(offset: int, *, rows: int, k_total: int) -> tuple[int, int]:
    """``byte offset -> (row, first k of the block it scales)``. Inverse of :func:`sf_byte_offset`.

    Only ``(row, sf_block)`` is recoverable — 16 K-elements share the byte, which is the format —
    so the returned ``k`` is the block's first element. ``sf_byte_offset(row, k) == offset`` holds
    for the returned pair and for every other ``k`` in the same block.
    """
    total = sf_tensor_bytes(rows, k_total)
    if not 0 <= offset < total:
        raise ValueError(f"offset {offset} outside the {total}-byte scale tensor")
    _, k_atoms = sf_atom_grid(rows, k_total)
    atom, within = divmod(offset, ATOM_BYTES)
    m_atom, k_atom = divmod(atom, k_atoms)
    row_in_atom, sf_block_in_atom = atom_offset_inverse(within)
    return m_atom * ATOM_ROWS + row_in_atom, k_atom * ATOM_K + sf_block_in_atom * SF_VEC_SIZE


# =============================================================================================
# Quantization arithmetic — what a tolerance can honestly be derived from
# =============================================================================================

#: Every finite E2M1 magnitude, ascending: the grid one element is rounded onto. Codes 0..7 are the
#: positive half; bit 3 is the sign.
E2M1_MAGNITUDES: tuple[float, ...] = tuple(FP4_E2M1.decode(c) for c in range(8))

#: 6.0 — the divisor in ``scale = amax / E2M1_MAX``. Not a tunable: it is the largest number the
#: element format holds, so any larger ratio saturates and the error bound below stops being true.
E2M1_MAX = E2M1_MAGNITUDES[-1]

#: Half the largest *relative* gap in E2M1's normal range. Within a binade the two representable
#: magnitudes are ``2^e`` and ``1.5 * 2^e``: the step is ``0.5 * 2^e`` and round-to-nearest is off
#: by at most ``0.25 * 2^e``, i.e. a quarter of the smaller neighbour. This is the unit roundoff of
#: a two-significand-bit format, ``2^-2``, and it is the entire numerical content of "e2m1 has one
#: mantissa bit". Every tolerance the rung can honestly claim is built from it — but the constant
#: in the correctness gate is Huy's to argue, and this module deliberately exports no tolerance.
E2M1_UNIT_ROUNDOFF = 0.25

#: ``2^-3`` — the largest relative *overshoot* of a round-UP onto the UE4M3 grid in its normal
#: range (step ``2^(e-3)`` over a value ``>= 2^e``). It is why :func:`scale_for_amax` can promise
#: ``amax/scale >= 6/(1+2^-3) = 5.33``, hence that the block's largest element lands on 6.0.
UE4M3_CEIL_RELATIVE = 0.125

_UE4M3_ASCENDING: tuple[tuple[float, int], ...] = tuple(
    sorted(
        ((v, c) for c, v in enumerate(SF_UE4M3.values()) if math.isfinite(v)), key=lambda t: t[0]
    )
)

#: Largest and smallest positive representable scale, derived rather than typed: 448 and 2^-9.
UE4M3_MAX = _UE4M3_ASCENDING[-1][0]
UE4M3_MIN_POSITIVE = next(v for v, _ in _UE4M3_ASCENDING if v > 0.0)

#: ``2^-6`` — below this a UE4M3 scale is subnormal, its grid is absolute rather than relative, and
#: the ``amax/scale >= 5.33`` argument fails. Both the tightness and the idempotence properties of
#: :func:`scale_for_amax` are claimed only above it.
UE4M3_MIN_NORMAL = math.ldexp(1.0, 1 - SF_UE4M3.bias)


def ue4m3_ceil_code(value: float) -> int:
    """Smallest UE4M3 code whose value is ``>= value``. Rounds **up**, deliberately.

    Round-to-nearest is the wrong rule for a block scale, and the reason is one line of algebra:
    the quantizer divides by the scale, so a scale that rounds *down* puts ``amax / scale`` above
    E2M1's maximum of 6, the block's largest element saturates, and the error on exactly the
    element that mattered most is unbounded. Rounding up costs at most
    :data:`UE4M3_CEIL_RELATIVE` of dynamic range and makes ``|x / scale| <= 6`` a theorem rather
    than a hope.

    Raises rather than clamping above 448: a block whose ``amax/6`` exceeds the scale format's
    maximum cannot be represented at all, and quietly clamping would hand back a silently saturated
    tile. What to do about it belongs to the caller, not to the encoder.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"ue4m3_ceil_code: {value} is not a finite non-negative magnitude")
    for v, c in _UE4M3_ASCENDING:
        if v >= value:
            return c
    raise ValueError(
        f"ue4m3_ceil_code: {value} exceeds UE4M3's maximum {UE4M3_MAX:g}; this block's "
        f"amax/{E2M1_MAX:g} is unrepresentable — rescale the operand before quantizing"
    )


def scale_for_amax(amax: float) -> tuple[int, float]:
    """``amax -> (ue4m3 code, its exact value)`` for one block. ``amax == 0`` gives the zero code.

    ``scale = ceil_ue4m3(amax / 6)``. Two properties follow, and both are asserted in the tests:

    * **No saturation, always.** ``scale >= amax/6``, so every element satisfies
      ``|x/scale| <= 6`` and the round-trip error is bounded by
      ``E2M1_UNIT_ROUNDOFF * max(|x|, scale)``.
    * **Tightness, when the scale is normal.** ``scale <= (amax/6) * (1 + 2^-3)``, so
      ``amax/scale >= 16/3 = 5.33``, which rounds to 6.0 — the block's largest element reaches the
      top of E2M1's grid instead of wasting a binade, and quantize-dequantize-quantize is a fixed
      point. Below :data:`UE4M3_MIN_NORMAL` the UE4M3 grid is absolute, that ratio argument fails,
      and only the first property survives.
    """
    if not math.isfinite(amax) or amax < 0:
        raise ValueError(
            f"scale_for_amax: amax must be a finite non-negative magnitude, got {amax}"
        )
    if amax == 0.0:
        return 0, 0.0
    code = ue4m3_ceil_code(amax / E2M1_MAX)
    return code, SF_UE4M3.decode(code)


def e2m1_round_code(value: float) -> int:
    """Round one already-scaled value onto the E2M1 grid: nearest, ties to even, saturating at ±6.

    Ties-to-even is not decoration here. E2M1's grid is coarse enough that ``|y| = 5`` — the exact
    midpoint of the 4-to-6 gap — is a value real blocks land on, and rounding every such tie upward
    would bias each quantized tensor away from zero. Even mantissa wins, so 5 -> 4 and 0.75 -> 1.0.
    """
    if not math.isfinite(value):
        raise ValueError(
            "e2m1_round_code: E2M1 encodes neither NaN nor Inf (NanInfEncoding::NONE) — a "
            "non-finite input cannot be represented and must be rejected, not saturated"
        )
    code = 8 if value < 0 else 0
    a = abs(value)
    best = 0
    for i in range(1, 8):
        lo, hi = E2M1_MAGNITUDES[i - 1], E2M1_MAGNITUDES[i]
        if a >= hi:
            best = i
            continue
        mid = 0.5 * (lo + hi)
        # The list index IS the (exponent, mantissa) code, so an even index has mantissa 0 and wins
        # the tie. Non-ties take whichever side of the midpoint they fall on.
        if a > mid or (a == mid and i % 2 == 0):
            best = i
        break
    return code | best
