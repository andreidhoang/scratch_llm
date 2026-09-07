"""K2/A-R1 — the tuning layer around the FA2-Triton kernel: arch-keyed configs, causal block
skipping, an exp2 path, and a two-kernel backward.

``fa2.py`` is the A2.1 port: one config list for every architecture, `tl.exp`, a single mask
expression applied to every key tile, and a backward that atomically accumulates dQ from the dK/dV
kernel. It reaches ~50% of SDPA on sm_120 and its backward does not launch there at all
(SIXTY_DAYS_SIX_LADDERS.md §03). This rung leaves that module untouched and adds, beside it, the
four things the plan row names — every one of which is a decision *about* the kernel rather than
arithmetic *inside* it:

  1. **Arch-keyed autotune.** ``fa2.py:27-32`` offers one list of 32 configs to every device.
     ``BLOCK_Q x BLOCK_K x num_stages`` is a shared-memory budget, and Hopper's per-CTA cap is
     227 KB against client Blackwell's 99 KB — so a config list that is merely large on H100 is a
     launch failure on an RTX PRO. :func:`configs_for` computes each candidate's bytes and asks
     ``hopper_contracts.check_smem_budget`` whether they fit, per arch. The filter is ARITHMETIC,
     not a hand-kept per-arch list, which is the difference between fixing this overflow and fixing
     the one config that happened to be reported.
  2. **Causal block skipping.** ``fa2.py:82`` already stops the k-loop at the diagonal; it still
     evaluates ``tl.where`` on every visited tile. :func:`kv_span` splits the visit into the blocks
     that are wholly below the diagonal (no mask at all) and the one block that straddles it.
  3. **exp2 with a log2e-prescaled scale.** :func:`softmax_scale` folds ``log2(e)`` into the scale
     on the host so the kernel can use the hardware's ``ex2.approx`` and never pay for a
     multiply-then-exp. :func:`lse_from_running_state` converts back, because everything
     downstream — this repo's oracle, both backward kernels — reads LSE in NATURAL log units.
  4. **A split backward.** :func:`bwd_grids` gives the two grids: one program per KV block
     accumulating dK/dV in registers, one program per query block accumulating dQ in registers.
     ``fa2.py:293`` instead does ``tl.atomic_add`` into global dQ from inside the dK/dV loop.

Nothing here imports torch's GPU stack, triton, or ``fa2``: the point of the split between this
module and ``fa2_tuned_triton`` is that every decision above is decidable on a laptop and is tested
there. The Triton kernels — and the one ``# HUY:`` hole — live in ``fa2_tuned_triton.py``, imported
lazily by the entry points at the bottom.

Spec, floor and kill rule: ``experiments/K2/A-R1/spec.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from scratch_llm.kernels.common.hopper_contracts import (
    ArchLimits,
    arch_for,
    check_smem_budget,
    max_stages,
)

#: The K2 rung identity, in one place, so the wrapper, the tests, the bench and the spec cannot
#: drift apart. ``SOURCE`` is the module that carries the hole, repo-relative — the same string
#: ``tests/conftest.py``'s ``@pytest.mark.hole`` marker is given.
RUNG = "K2/A-R1"
SOURCE = "src/scratch_llm/kernels/attention/prefill/fa2_tuned_triton.py"
BASELINE = "src/scratch_llm/kernels/attention/prefill/fa2.py"

#: 16-bit operands. Not a preference: the whole shared-memory model below counts two bytes per
#: element, and ``tl.dot`` reaches the tensor cores through a 16-bit A/B operand. Handing this
#: kernel fp32 q/k/v would double every number :func:`configs_for` reasons about while leaving the
#: chosen config unchanged — a silent 2x under-count of shared memory, i.e. exactly the class of
#: bug this rung exists to remove. :func:`unsupported_reason` rejects it instead.
ELEM_BYTES = 2

#: ``log2(e)`` and ``ln(2)``. The exp2 path lives or dies on using them in the right direction:
#: the scale is multiplied by LOG2E going in, and the running max is multiplied by LN2 coming out.
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


# =============================================================================================
# Configs — one candidate set, filtered per arch by arithmetic
# =============================================================================================


@dataclass(frozen=True)
class FA2Config:
    """One autotune candidate. The four numbers ``triton.Config`` is built from."""

    block_q: int
    block_k: int
    num_warps: int
    num_stages: int

    def as_kwargs(self) -> dict[str, int]:
        """The ``triton.Config`` meta-parameter dict (the two that are ``tl.constexpr``)."""
        return {"BLOCK_Q": self.block_q, "BLOCK_K": self.block_k}


#: Which of the three kernels a shared-memory model is for. They differ in which tiles are
#: RESIDENT for the program's lifetime and which are re-loaded (and therefore multi-buffered by
#: Triton's pipeliner) every loop iteration — and that, not the tile area, is what decides the
#: budget.
Kind = Literal["fwd", "bwd_dkdv", "bwd_dq"]

#: The candidate set: ``fa2.py:27-32``'s grid, plus a 4-stage row. The extra row is not
#: speculative tuning — it is there so the filter has something to reject on HOPPER too, which is
#: what keeps ``configs_for`` honest as an arithmetic budget rather than an sm_120-shaped
#: special case. A test asserts this is a superset of ``fa2.py``'s grid: the tuned selector must
#: only ever REMOVE configs the untuned kernel already offered, never invent one.
CANDIDATES: tuple[FA2Config, ...] = tuple(
    FA2Config(bq, bk, w, s)
    for bq, bk in ((64, 64), (128, 64), (64, 128), (128, 128))
    for w in (4, 8)
    for s in (2, 3, 4)
)


def smem_terms(
    cfg: FA2Config, d: int, *, kind: Kind, elem_bytes: int = ELEM_BYTES
) -> tuple[int, int]:
    """``(bytes_per_stage, resident_bytes)`` for one config — the shared-memory model.

    Term-for-term from upstream's own sm_120 admission checks, which compute exactly this and
    compare it against the arch's capacity rather than keeping a list of blessed tile shapes:

      * forward — ``oss/flash-attention/flash_attn/cute/flash_fwd_sm120.py:48-54``::

            smem_usage_Q = tile_m * head_dim * 2                        # resident
            smem_usage_K = tile_n * head_dim   * num_stages * 2         # staged
            smem_usage_V = tile_n * head_dim_v * num_stages * 2         # staged

      * dK/dV backward — ``flash_bwd_sm120.py:43-50``, which inverts the residency: K and V are
        the program's own block and stay, Q and dO stream past and are staged.

    ``bwd_dq`` has no upstream line to quote; it is the forward's residency plus dO, derived by
    the same rule. It is called out here rather than smuggled in, because a model term that was
    guessed and one that was quoted are different kinds of claim.

    This is a MODEL. Triton, not this function, decides the real allocation, and the ncu number on
    the box is the arbiter. It is deliberately the conservative direction: it counts every stage
    of every streamed tile, so a config it admits cannot overflow because of a term left out.
    """
    q = cfg.block_q * d * elem_bytes
    kv = cfg.block_k * d * elem_bytes
    if kind == "fwd":
        return 2 * kv, q  # K and V staged; Q loaded once and held
    if kind == "bwd_dkdv":
        return 2 * q, 2 * kv  # Q and dO stream past; K and V are this program's own block
    if kind == "bwd_dq":
        return 2 * kv, 2 * q  # K and V stream past; Q and dO are this program's own block
    raise ValueError(f"unknown kernel kind {kind!r}; expected one of fwd, bwd_dkdv, bwd_dq")


def smem_bytes(cfg: FA2Config, d: int, *, kind: Kind, elem_bytes: int = ELEM_BYTES) -> int:
    """Total modelled shared memory for one config — what the arch cap is compared against."""
    per_stage, resident = smem_terms(cfg, d, kind=kind, elem_bytes=elem_bytes)
    return per_stage * cfg.num_stages + resident


def smem_violations(
    cfg: FA2Config, d: int, arch: ArchLimits, *, kind: Kind, elem_bytes: int = ELEM_BYTES
) -> list[str]:
    """``check_smem_budget``'s verdict for one config, verbatim — including the opt-in note."""
    per_stage, resident = smem_terms(cfg, d, kind=kind, elem_bytes=elem_bytes)
    return check_smem_budget(
        bytes_per_stage=per_stage, stages=cfg.num_stages, arch=arch, extra_bytes=resident
    )


def fits(
    cfg: FA2Config, d: int, arch: ArchLimits, *, kind: Kind, elem_bytes: int = ELEM_BYTES
) -> bool:
    """True iff this config is launchable on ``arch``.

    ``check_smem_budget`` reports two different things in one list, and only one of them is fatal.
    Exceeding ``smem_per_cta`` is a hard stop. Exceeding the 48 KB opt-in threshold is not: it is
    legal, and the launcher must call ``cudaFuncSetAttribute`` for it. Every config this kernel
    would want is over 48 KB, so treating the opt-in note as a failure would leave the arch with
    no configs at all. Triton's launcher raises that attribute itself at launch time — a runtime
    property of the toolchain, not something this repo can cite a line for, so a config admitted
    here that then fails to launch means the toolchain changed and this comment is the first place
    to look.
    """
    return all(
        e.startswith("OPT-IN REQUIRED")
        for e in smem_violations(cfg, d, arch, kind=kind, elem_bytes=elem_bytes)
    )


def configs_for(
    arch: ArchLimits,
    d: int,
    *,
    kind: Kind = "fwd",
    elem_bytes: int = ELEM_BYTES,
    candidates: tuple[FA2Config, ...] = CANDIDATES,
) -> list[FA2Config]:
    """The candidates that fit ``arch`` at head dim ``d`` — the fix for the sm_120 overflow.

    Ordered smallest-first so that if autotune is ever short-circuited (a single-config debug run,
    a compile-time budget) the config it lands on is the one most likely to launch.

    An empty result is not silently tolerated by the caller: :func:`require_configs` turns it into
    a message that names the arch, the head dim, and the smallest candidate's byte count, because
    "no config fits" at 03:00 on a rented box is otherwise indistinguishable from "autotune hung".
    """
    fitting = [c for c in candidates if fits(c, d, arch, kind=kind, elem_bytes=elem_bytes)]
    return sorted(
        fitting, key=lambda c: (smem_bytes(c, d, kind=kind, elem_bytes=elem_bytes), c.num_warps)
    )


def configs_for_cc(
    cc: tuple[int, int],
    d: int,
    *,
    kind: Kind = "fwd",
    elem_bytes: int = ELEM_BYTES,
    candidates: tuple[FA2Config, ...] = CANDIDATES,
) -> list[FA2Config]:
    """:func:`configs_for` keyed on a compute capability, e.g. ``(12, 0)``.

    ``arch_for`` raises ``KeyError`` for a capability that is not in ``hopper_contracts.ARCH``
    (sm_80, sm_86, sm_89 today). That is deliberate and is left to propagate: this rung's exit
    number is stated on H100, and a silent fallback to "assume 99 KB" would let an untested arch
    produce a number that looks like the rung's. Adding a row to ``ARCH`` with a citation is the
    supported way to run here on another card.
    """
    return configs_for(arch_for(cc), d, kind=kind, elem_bytes=elem_bytes, candidates=candidates)


def require_configs(
    cc: tuple[int, int], d: int, *, kind: Kind = "fwd", elem_bytes: int = ELEM_BYTES
) -> list[FA2Config]:
    """:func:`configs_for_cc`, raising a diagnosis rather than returning an empty list."""
    chosen = configs_for_cc(cc, d, kind=kind, elem_bytes=elem_bytes)
    if chosen:
        return chosen
    arch = arch_for(cc)
    smallest = min(CANDIDATES, key=lambda c: smem_bytes(c, d, kind=kind, elem_bytes=elem_bytes))
    per_stage, resident = smem_terms(smallest, d, kind=kind, elem_bytes=elem_bytes)
    raise ValueError(
        f"no {kind} config fits {arch.name} at head dim {d}: the smallest candidate "
        f"{smallest} needs {smem_bytes(smallest, d, kind=kind, elem_bytes=elem_bytes)} B against a "
        f"{arch.smem_per_cta} B cap. At {per_stage} B/stage plus {resident} B resident this arch "
        f"affords {max_stages(bytes_per_stage=per_stage, arch=arch, extra_bytes=resident)} stages — "
        f"shrink BLOCK_K, or split the head dim, before adding a smaller candidate."
    )


# =============================================================================================
# Which blocks to visit — causal block skipping with the mask on the diagonal only
# =============================================================================================


@dataclass(frozen=True)
class KVSpan:
    """For one query block: the KV columns to visit, split at the diagonal.

    ``[0, full_end)`` are blocks every query in the tile can see in full — no causal mask, and no
    key-bounds mask either, since ``full_end`` is a block-aligned index at or below ``n_ctx``.
    ``[full_end, masked_end)`` is the diagonal: at most one block for ``BLOCK_K >= BLOCK_Q``, and
    it is the only one that pays for a ``tl.where``. Beyond ``masked_end`` every key is in the
    future of every query in the tile and the block is never loaded.
    """

    full_end: int
    masked_end: int

    @property
    def masked_span(self) -> int:
        """Keys covered by the masked phase, in ELEMENTS — one BLOCK_K per straddling block."""
        return max(0, self.masked_end - self.full_end)


def kv_span(*, q_start: int, block_q: int, block_k: int, n_ctx: int, causal: bool) -> KVSpan:
    """The KV visit plan for the query block starting at ``q_start``.

    ``full_end = (q_start // block_k) * block_k`` is exact, not a conservative rounding: a KV block
    at column ``c`` is wholly visible to the tile iff ``c + block_k - 1 <= q_start`` (its LAST key
    is at or before the tile's FIRST query), and for power-of-two block sizes the largest such
    block-aligned ``c`` is precisely ``(q_start // block_k) * block_k - block_k``.

    The same boundary appears upstream as
    ``oss/flash-attention/flash_attn/cute/block_info.py:118-134``
    (``get_n_block_min_causal_local_mask``: "if we have separate iterations with causal or local
    masking ... where do we stop", returning ``n_idx // tile_n``). vLLM's Triton prefill kernel
    stops one step short of this: ``oss/vllm/vllm/v1/attention/ops/triton_prefill_attention.py:122``
    clamps the loop bound to the diagonal but then evaluates ``pos_q >= pos_k`` on every tile it
    does visit (``:144``), so the mask cost stays O(S²/2) instead of O(S).

    Non-causal is the same shape with a different reason: the full region is every whole block,
    and the "masked" region is the ragged tail, which needs a key-BOUNDS mask and no causal one.
    """
    if block_q <= 0 or block_k <= 0 or n_ctx <= 0:
        raise ValueError(
            f"block_q={block_q}, block_k={block_k}, n_ctx={n_ctx} must all be positive"
        )
    if not causal:
        return KVSpan(full_end=(n_ctx // block_k) * block_k, masked_end=n_ctx)
    return KVSpan(full_end=(q_start // block_k) * block_k, masked_end=min(n_ctx, q_start + block_q))


@dataclass(frozen=True)
class QSpan:
    """For one KV block, in the dK/dV backward: the query rows to visit, split at the diagonal.

    The mirror of :class:`KVSpan` with the roles exchanged. ``[0, masked_start)`` is skipped —
    those queries are entirely before this block's keys, so every entry is masked and dK/dV get
    nothing from them. ``[masked_start, full_start)`` straddles the diagonal. ``[full_start,
    n_ctx)`` is unmasked by causality, though the LAST block there may still be ragged, so unlike
    the forward's full region it keeps its row-bounds mask.
    """

    masked_start: int
    full_start: int


def q_span(*, kv_start: int, block_q: int, block_k: int, n_ctx: int, causal: bool) -> QSpan:
    """The query visit plan for the KV block starting at ``kv_start`` (dK/dV backward).

    ``masked_start = (kv_start // block_q) * block_q``: a query block at row ``r`` contributes iff
    its LAST query is at or after this block's FIRST key, ``r + block_q - 1 >= kv_start``.
    ``full_start = ceil((kv_start + block_k - 1) / block_q) * block_q``: a query block is wholly
    unmasked iff its FIRST query is at or after this block's LAST key.

    This is what makes the split backward pay: without it the dK/dV kernel walks all of Q for
    every KV block (``fa2.py:243`` starts at ``q_start`` but then masks every tile), and the dQ
    kernel would walk all of K for every query block.
    """
    if block_q <= 0 or block_k <= 0 or n_ctx <= 0:
        raise ValueError(
            f"block_q={block_q}, block_k={block_k}, n_ctx={n_ctx} must all be positive"
        )
    if not causal:
        # No split: with no causal mask there is nothing for the masked phase to do, and the
        # ragged LAST query block still needs its row-bounds mask, which lives inside the block
        # body either way. _fa2_tuned_bwd_dkdv_kernel does exactly this.
        return QSpan(masked_start=0, full_start=0)
    masked_start = (kv_start // block_q) * block_q
    full_start = -(-(kv_start + block_k - 1) // block_q) * block_q
    return QSpan(masked_start=masked_start, full_start=min(full_start, n_ctx))


# =============================================================================================
# exp2 and the LSE units
# =============================================================================================


def softmax_scale(d: int, *, use_exp2: bool, scale: float | None = None) -> float:
    """The scale handed to the kernel: ``1/sqrt(d)``, times ``log2(e)`` on the exp2 path.

    Prescaling on the host is the whole trick — the kernel then computes ``exp2(qk - m)`` with the
    hardware's ``ex2.approx`` and never issues the ``* log2(e)`` that ``tl.exp`` lowers to.
    ``oss/vllm/vllm/v1/attention/ops/triton_prefill_attention.py:244`` does exactly this
    (``sm_scale *= RCP_LN2``, commented "rescale with 1/ln(2) for triton exp2"), and its kernel
    then calls ``tl.math.exp2`` at ``:175`` and ``:179``.
    """
    base = 1.0 / math.sqrt(d) if scale is None else scale
    return base * LOG2E if use_exp2 else base


def lse_from_running_state(m_i, l_i, *, use_exp2: bool):  # noqa: ANN001 - float or Tensor
    """``logsumexp`` in NATURAL log units from the kernel's running max and denominator.

    On the exp2 path ``m_i`` is in log2 units (it is the max of a log2-domain score) while ``l_i``
    is dimensionless — ``sum(exp2(s2 - m2)) == sum(exp(s - m))`` exactly — so only ``m_i`` gets
    converted. Getting this wrong does not produce a wrong forward output: ``O = acc / l_i`` never
    touches ``m_i``. It produces a forward that passes its own test and a backward that is wrong
    by a factor of ``exp((1 - ln 2) * m)`` per row, which is why this conversion is tested on its
    own rather than only through the kernel.

    Both backward kernels and ``kernels/attention/reference.py`` consume natural-log LSE, so the
    flag must not leak past this function.
    """
    m = m_i * LN2 if use_exp2 else m_i
    return m + (torch.log(l_i) if isinstance(l_i, Tensor) else math.log(l_i))


# =============================================================================================
# The backward split
# =============================================================================================


@dataclass(frozen=True)
class BwdGrids:
    """The two launch grids of the split backward, and the shape of what each program owns."""

    dkdv: tuple[int, int]
    dq: tuple[int, int]


def bwd_grids(*, n_ctx: int, batch: int, dkdv: FA2Config, dq: FA2Config) -> BwdGrids:
    """Grids for the dK/dV kernel and the dQ kernel.

    Two kernels, not one, and no atomics in either: each program owns its output block outright
    and accumulates it in registers, so dK/dV and dQ are each written exactly once.
    ``fa2.py:293`` instead does ``tl.atomic_add`` into global dQ from inside the dK/dV loop —
    every query block's dQ is read-modify-written once per KV block it sees, which is O(S/BLOCK_K)
    global atomics per element and forces dQ to be zero-initialised before the launch
    (``fa2.py:332``). Upstream splits it the same way this does: the sm_100 backward ships as
    ``oss/flash-attention/flash_attn/cute/flash_bwd_mla_dk_sm100.py`` and
    ``flash_bwd_mla_dq_dqv_sm100.py``, two kernels either side of
    ``flash_bwd_preprocess.py``'s D-vector pass.

    The two kernels get SEPARATE configs on purpose. Their residency is inverted — dK/dV holds K
    and V and streams Q and dO, dQ holds Q and dO and streams K and V — so on a 99 KB arch they do
    not admit the same block shapes, and forcing one config on both is how the untuned backward
    ends up asking for a tile that only Hopper can hold.
    """
    return BwdGrids(
        dkdv=(-(-n_ctx // dkdv.block_k), batch),
        dq=(-(-n_ctx // dq.block_q), batch),
    )


# =============================================================================================
# Entry points. Everything below needs a GPU; everything above does not.
# =============================================================================================


def unsupported_reason(q: Tensor, k: Tensor, v: Tensor) -> str | None:
    """Why this problem cannot run on this rung, or ``None`` if it can.

    Every rule here is a property of what was actually built, not a corner cut quietly:

    * **Self-attention only** — same as ``fa2.py:130-131``.
    * **Matched leading dims (no GQA)** — the wrapper flattens all leading dims into one batch
      axis, so a KV head cannot be shared by several query heads. Expressing GQA needs a query-head
      to KV-head index map in the kernel, which is A-R3's shape, not this rung's. The bench Rung
      row therefore does not list ``gqa8k``; a rung must not advertise a shape its wrapper rejects.
    * **16-bit operands** — see :data:`ELEM_BYTES`.
    * **Power-of-two head dim, 16..128** — ``tl.arange`` needs a power of two, and past 128 no
      candidate config fits any arch in :data:`CANDIDATES`.
    """
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        return (
            f"leading dims must match (no GQA at this rung): q{tuple(q.shape[:-2])} "
            f"k{tuple(k.shape[:-2])} v{tuple(v.shape[:-2])}"
        )
    if k.shape[-2] != q.shape[-2] or v.shape[-2] != q.shape[-2]:
        return (
            f"kernel assumes self-attention (key len {k.shape[-2]}, value len {v.shape[-2]} != "
            f"query len {q.shape[-2]})"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return f"q, k, v must share a dtype; got {q.dtype}, {k.dtype}, {v.dtype}"
    if q.element_size() != ELEM_BYTES:
        return (
            f"{q.dtype} is {q.element_size()} bytes/element but the shared-memory model that picks "
            f"the config assumes {ELEM_BYTES}; pass bfloat16 or float16"
        )
    d = q.shape[-1]
    if d != k.shape[-1] or d != v.shape[-1]:
        return f"head dims disagree: q {d}, k {k.shape[-1]}, v {v.shape[-1]}"
    if d & (d - 1) or not 16 <= d <= 128:
        return (
            f"head dim must be a power of two in 16..128 (tl.arange, and the smem budget); got {d}"
        )
    return None


def _check(q: Tensor, k: Tensor, v: Tensor, *, fn: str) -> None:
    why = unsupported_reason(q, k, v)
    if why is not None:
        raise ValueError(f"{fn}: {why}")


def fa2_tuned_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
    use_exp2: bool = True,
    skip_masked_blocks: bool = True,
) -> tuple[Tensor, Tensor]:
    """Tuned FA2 forward. ``q,k,v``: ``(..., N, d)``; returns ``(O, L)``, ``L`` in natural log.

    Signature-compatible with :func:`~scratch_llm.kernels.attention.prefill.fa2.
    flash_attention_triton_forward` so the two can be swapped in one bench and the delta is the
    tuning and nothing else. The two flags exist to be turned OFF: A-R1's number is only
    interpretable if each lever can be measured alone, and a flag that cannot be disabled is a
    claim that cannot be falsified.
    """
    _check(q, k, v, fn="fa2_tuned_forward")
    from scratch_llm.kernels.attention.prefill import fa2_tuned_triton as kern

    return kern.forward(
        q, k, v, is_causal=is_causal, use_exp2=use_exp2, skip_masked_blocks=skip_masked_blocks
    )


def fa2_tuned_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    lse: Tensor,
    do: Tensor,
    *,
    is_causal: bool = False,
    use_exp2: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split backward: one dK/dV kernel, one dQ kernel, no atomics. ``lse`` in natural log."""
    _check(q, k, v, fn="fa2_tuned_backward")
    from scratch_llm.kernels.attention.prefill import fa2_tuned_triton as kern

    return kern.backward(q, k, v, o, lse, do, is_causal=is_causal, use_exp2=use_exp2)


def baseline_forward(q: Tensor, k: Tensor, v: Tensor, *, is_causal: bool = False):  # noqa: ANN201
    """``fa2.py``'s untuned kernel, for the A/B this rung's number is a statement about.

    Imported lazily and only here: ``fa2`` imports Triton at module scope (``fa2.py:23``), so a
    top-level import would make this module — and every CPU test of the arithmetic above —
    unimportable on a box without a GPU.
    """
    from scratch_llm.kernels.attention.prefill.fa2 import flash_attention_triton_forward

    return flash_attention_triton_forward(q, k, v, is_causal=is_causal)
