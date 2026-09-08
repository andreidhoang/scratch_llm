"""K2/A-R1 — FA2-Triton tuned for Hopper. Two tiers of gate, in ascending cost.

  CPU  everything this rung actually decides: which configs an arch can afford (and that the
       sm_120 overflow is gone), which KV blocks a query tile visits, that exp2 with a
       log2e-prescaled scale is the same function as exp, that LSE leaves in natural log units,
       and how the backward splits. Milliseconds, here, no GPU, no Triton.
  gpu  correctness against this repo's own FA2 oracle, and against the untuned kernel this rung
       is a delta on. Needs a CUDA device with Triton.

There is no drydock tier and there cannot be one. A ``.cu`` is compiled by nvcc, which runs in a
container on this laptop, so its SASS and its ptxas report can be asserted on before anything is
rented. A Triton kernel is compiled by the driver, on the device, at first call — there is no
artifact to inspect here. Everything the dry dock would have caught for this rung is arithmetic
(the shared-memory budget, the block plan, the scale units) and is caught by the CPU tier below
from published constants instead. That is a weaker check than reading real SASS, and the honest
statement of it is: this suite proves the DECISIONS are right and proves nothing about the
generated code.

Spec: experiments/K2/A-R1/spec.md
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import torch

from scratch_llm.kernels.attention.prefill.fa2_tuned import (
    BASELINE,
    CANDIDATES,
    ELEM_BYTES,
    LN2,
    LOG2E,
    RUNG,
    SOURCE,
    FA2Config,
    Kind,
    bwd_grids,
    configs_for,
    configs_for_cc,
    kv_span,
    lse_from_running_state,
    q_span,
    require_configs,
    smem_bytes,
    smem_terms,
    softmax_scale,
    unsupported_reason,
)
from scratch_llm.kernels.attention.reference import (
    FlashAttentionPyTorch,
    flash_attention_forward,
)
from scratch_llm.kernels.common.hopper_contracts import ARCH

_REPO = Path(__file__).resolve().parents[3]

#: The three kernels whose shared memory is modelled separately, because their residency differs.
#: Typed as ``Kind`` so a fourth entry that ``fa2_tuned`` does not model is a type error here
#: rather than a ``KeyError`` inside a parametrized run.
KINDS: tuple[Kind, ...] = ("fwd", "bwd_dkdv", "bwd_dq")

#: The head dim every K2 spec shape uses (bench/kernels/attention/k2_ladder.py SHAPES). Every
#: budget assertion below is stated at this width, because "which configs fit" is a statement
#: about a head dim and is meaningless without one.
D = 128

# ---------------------------------------------------------------------------------------------
# The tolerances. HUY SETS THESE, with their arguments, before the first measured run.
#
# Both are max-abs bounds against the fp32 oracle, and they are NOT the same number. The forward
# is one bf16 dot chain per query row plus an fp32 online softmax; the backward is three more dot
# chains whose A-operands (P and dS) are ROUNDED BACK to bf16 before each one — see
# fa2_tuned_triton._bwd_dkdv_block — so its error is larger and grows with how many query blocks a
# KV block sees, i.e. with sequence length and with causality.
#
# Deriving those two bounds, and saying which of the two rounding steps dominates, IS the rung's
# numerics lesson. Leaving them None is deliberate; _require_tolerance turns a forgotten one into
# a message rather than a green test.
# ---------------------------------------------------------------------------------------------
TOL_FWD: float | None = None
TOL_BWD: float | None = None


def _hole_is_open() -> bool:
    """True while the rescale core still carries the sentinel (same rule as tests/conftest.py)."""
    # Regex across whitespace, not a fixed substring: `ruff format` wraps a long
    # `raise NotImplementedError("HUY: ...")` onto two lines, which deletes that substring and makes
    # an unwritten kernel report as CLOSED — the one failure this convention exists to prevent.
    # Same rule as tests/conftest.py::hole_is_open.
    text = (_REPO / SOURCE).read_text(encoding="utf-8")
    return re.search(r'NotImplementedError\(\s*(?:#[^\n]*\n\s*)?"HUY:', text) is not None


def _tuned_source() -> str:
    return (_REPO / SOURCE).read_text(encoding="utf-8")


def _tuned_code() -> str:
    """The module's CODE, with every docstring and comment removed.

    The three assertions below search the kernel source for things that must (or must not) be
    there, and the module's own prose talks at length ABOUT atomics and about ``fa2.py``'s
    scales — so a substring search over the raw file finds the commentary and reports the
    opposite of the truth. ``ast.unparse`` drops comments on its own; the docstrings are stripped
    explicitly.
    """
    tree = ast.parse(_tuned_source())
    holder = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, holder) and ast.get_docstring(node, clean=False) is not None:
            node.body = node.body[1:]
    return ast.unparse(tree)


def _require_tolerance(which: str, value: float | None) -> float:
    if value is None:
        pytest.fail(
            f"{which} is unset. The correctness gate cannot run without a tolerance, and choosing "
            f"one is this rung's numerics lesson: set it in this file and write the one-line "
            f"argument into experiments/K2/A-R1/spec.md's 'Correctness gate' line before the first "
            f"measured run."
        )
    return value


# =============================================================================================
# The hole guard — exactly one per hole (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K2/A-R1", "src/scratch_llm/kernels/attention/prefill/fa2_tuned_triton.py")
def test_a_r1_rescale_core_is_filled_and_the_kernels_build() -> None:
    """Fails while ``_online_softmax_step`` raises; passes once it is written.

    A Triton rung has no compile step that runs on a laptop, so this guard asserts the only two
    things that need no silicon: that the body exists, and — where Triton is installed — that the
    module still executes to the end with it. That second half is weak on purpose and worth being
    honest about: ``@triton.jit`` does not parse a kernel body at decoration time, it parses at
    first compile, on the device. So this proves the module imports, not that the body is valid
    Triton. Correctness is ``test_forward_matches_the_oracle``, on the box.
    """
    assert not _hole_is_open(), (
        f"{RUNG} rescale core is still an open hole. Read experiments/K2/A-R1/spec.md, then write "
        f"the running max / denominator update and the O rescale in {SOURCE} "
        f"(_online_softmax_step). Everything around it — the block plan, the config filter, the "
        f"exp2 scale, both backward kernels — is already there."
    )
    pytest.importorskip("triton", reason="Triton absent here; the import check is box-only")
    from scratch_llm.kernels.attention.prefill import fa2_tuned_triton

    assert fa2_tuned_triton.forward is not None


# =============================================================================================
# CPU — the config filter. This is the fix for the sm_120 overflow, and it is pure arithmetic.
# =============================================================================================


def test_candidate_set_is_a_superset_of_the_untuned_kernels_grid() -> None:
    """The tuned selector may only REMOVE configs ``fa2.py`` already offered, never invent one.

    Parsed out of ``fa2.py`` rather than copied, so the day someone widens that grid this test
    fails instead of the two files quietly describing different kernels.
    """
    src = (_REPO / BASELINE).read_text(encoding="utf-8")
    pairs_m = re.search(r"for bq, bk in (\[.*?\])", src, re.S)
    warps_m = re.search(r"for w in (\(.*?\))", src)
    stages_m = re.search(r"for s in (\(.*?\))", src)
    # Named, not chained: a grid this parser cannot find means fa2.py's autotune loop was rewritten,
    # and that has to say so rather than die on `None.group(1)` three lines from the real cause.
    assert pairs_m is not None, f"{BASELINE} has no `for bq, bk in [...]` tile grid to parse"
    assert warps_m is not None, f"{BASELINE} has no `for w in (...)` num_warps grid to parse"
    assert stages_m is not None, f"{BASELINE} has no `for s in (...)` num_stages grid to parse"
    pairs = ast.literal_eval(pairs_m.group(1))
    warps = ast.literal_eval(warps_m.group(1))
    stages = ast.literal_eval(stages_m.group(1))
    baseline = {FA2Config(bq, bk, w, s) for bq, bk in pairs for w in warps for s in stages}
    missing = baseline - set(CANDIDATES)
    # `key=` because FA2Config is frozen but not ordered: bare sorted() would raise TypeError while
    # building the message, and eat the report of what is actually missing.
    assert not missing, (
        f"fa2.py offers configs this rung's candidate set does not: "
        f"{sorted(missing, key=lambda c: (c.block_q, c.block_k, c.num_warps, c.num_stages))}"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_every_config_offered_for_sm120_fits_its_99kb_cap(kind: Kind) -> None:
    """The plan row's headline: "arch-keyed autotune fixes sm120 smem overflow".

    99 KB per CTA is not a number this repo invented — ``hopper_contracts.ARCH['sm_120a']`` and
    upstream FlashAttention's own sm_120 admission check
    (``oss/flash-attention/flash_attn/cute/flash_fwd_sm120.py:56``, "SM120 has 99 KB shared memory")
    agree, and so does fla's ``Backend.ADA = 101376`` (``oss/fla/fla/utils/_device.py:197``),
    which is 99 KB to the byte.
    """
    cap = ARCH["sm_120a"].smem_per_cta
    assert cap == 99 * 1024
    chosen = configs_for_cc((12, 0), D, kind=kind)
    assert chosen, f"no {kind} config fits sm_120a at head dim {D} — the arch is unusable"
    for cfg in chosen:
        assert smem_bytes(cfg, D, kind=kind) <= cap, (
            f"{cfg} models {smem_bytes(cfg, D, kind=kind)} B"
        )


@pytest.mark.parametrize("kind", KINDS)
def test_a_hopper_config_exists_that_sm120_must_reject(kind: Kind) -> None:
    """The filter has to be doing work, not passing everything through.

    If every config Hopper accepts also fit client Blackwell there would be no overflow to fix and
    no reason for this module to exist. The assertion is the difference itself, not a hand-listed
    shape: something in the candidate set must be admissible at 227 KB and inadmissible at 99 KB.
    """
    hopper = set(configs_for(ARCH["sm_90a"], D, kind=kind))
    blackwell_client = set(configs_for(ARCH["sm_120a"], D, kind=kind))
    only_hopper = hopper - blackwell_client
    assert only_hopper, (
        f"every {kind} config Hopper accepts also fits sm_120a — then the sm_120 overflow this "
        f"rung fixes cannot be a shared-memory problem, and the diagnosis is wrong"
    )
    assert blackwell_client < hopper, "the smaller arch must admit a strict subset"


def test_the_untuned_backward_geometry_is_the_one_that_overflows_sm120() -> None:
    """The diagnosis, as arithmetic — the hypothesis the box is being rented to confirm.

    ``fa2.py:337-338,388`` launches the backward at BLOCK_Q = BLOCK_K = 64 with ``num_warps=4``
    and NO ``num_stages``, so Triton's own CUDA default applies (3 at time of writing — not
    citable from any checkout under oss/, which is why the box, not this test, settles it). At head dim 128 the dK/dV model —
    K and V resident, Q and dO staged, ``flash_bwd_sm120.py:43-50`` — puts that at 128 KB, over
    sm_120a's 99 KB cap and comfortably under Hopper's 227 KB. That is consistent with
    SIXTY_DAYS_SIX_LADDERS.md §03's "backward broken on sm120" and it is what makes the fix a
    STAGE COUNT rather than a tile change: the same geometry at two stages fits.

    This is a statement about the model, not a measurement. The spec's kill line says what happens
    if the box disagrees.
    """
    untuned = FA2Config(block_q=64, block_k=64, num_warps=4, num_stages=3)
    assert smem_bytes(untuned, D, kind="bwd_dkdv") > ARCH["sm_120a"].smem_per_cta
    assert smem_bytes(untuned, D, kind="bwd_dkdv") <= ARCH["sm_90a"].smem_per_cta
    two_stage = FA2Config(block_q=64, block_k=64, num_warps=4, num_stages=2)
    assert smem_bytes(two_stage, D, kind="bwd_dkdv") <= ARCH["sm_120a"].smem_per_cta
    assert two_stage in configs_for(ARCH["sm_120a"], D, kind="bwd_dkdv")


@pytest.mark.parametrize("arch_name", sorted(ARCH))
@pytest.mark.parametrize("kind", KINDS)
def test_no_offered_config_exceeds_the_arch_it_was_offered_for(arch_name: str, kind: Kind) -> None:
    """Every arch in the table, not just the two this rung is measured on."""
    arch = ARCH[arch_name]
    for cfg in configs_for(arch, D, kind=kind):
        assert smem_bytes(cfg, D, kind=kind) <= arch.smem_per_cta


def test_smem_model_terms_are_upstreams_terms() -> None:
    """The model, restated against the expressions it was taken from.

    Forward (``oss/flash-attention/flash_attn/cute/flash_fwd_sm120.py:48-54``): Q resident, K and
    V multiplied by ``num_stages``. dK/dV (``flash_bwd_sm120.py:43-50``): the residency inverted —
    K and V are the program's own block, Q and dO stream past. If this assertion ever has to be
    relaxed, the model has drifted from the thing it claims to be a model of.
    """
    cfg = FA2Config(block_q=128, block_k=64, num_warps=8, num_stages=3)
    per_stage, resident = smem_terms(cfg, D, kind="fwd")
    assert resident == cfg.block_q * D * ELEM_BYTES  # smem_usage_Q
    assert per_stage * cfg.num_stages == 2 * cfg.block_k * D * ELEM_BYTES * cfg.num_stages  # K + V
    per_stage, resident = smem_terms(cfg, D, kind="bwd_dkdv")
    assert resident == 2 * cfg.block_k * D * ELEM_BYTES  # smem_usage_K + smem_usage_V
    assert per_stage * cfg.num_stages == 2 * cfg.block_q * D * ELEM_BYTES * cfg.num_stages  # Q + dO


def test_an_arch_with_no_limits_row_is_named_not_guessed() -> None:
    """sm_80/86/89 are absent from ``hopper_contracts.ARCH`` and must stay a loud failure.

    A silent fallback to "assume 99 KB" would let an untested card produce a number that looks
    like this rung's exit number, against a floor measured somewhere else.
    """
    with pytest.raises(KeyError, match="add it to ARCH with a citation"):
        configs_for_cc((8, 0), D)


def test_no_fitting_config_is_a_diagnosis_not_an_empty_list() -> None:
    """ "nothing fits" at 03:00 on a rented box must not look like "autotune hung"."""
    with pytest.raises(ValueError, match=r"no fwd config fits sm_120a at head dim 256"):
        require_configs((12, 0), 256)


# =============================================================================================
# CPU — which blocks get visited. Brute-forced against the mask itself, not against a formula.
# =============================================================================================


def _visible(n_ctx: int, causal: bool) -> torch.Tensor:
    """``(n, n)`` bool: query ``i`` may attend key ``j``. The definition, with no blocking in it."""
    idx = torch.arange(n_ctx)
    return (idx[None, :] <= idx[:, None]) if causal else torch.ones(n_ctx, n_ctx, dtype=torch.bool)


_SPAN_CASES = [
    (64, 64, 512, True),
    (64, 64, 512, False),
    (128, 64, 512, True),  # BLOCK_Q > BLOCK_K: several diagonal blocks per query tile
    (64, 128, 512, True),  # BLOCK_Q < BLOCK_K: one, and it starts earlier
    (128, 128, 384, True),
    (64, 64, 500, True),  # n_ctx a multiple of neither
    (64, 64, 500, False),
    (128, 64, 300, True),
]


@pytest.mark.parametrize(("bq", "bk", "n_ctx", "causal"), _SPAN_CASES)
def test_kv_span_visits_exactly_the_blocks_that_contain_a_visible_key(
    bq: int, bk: int, n_ctx: int, causal: bool
) -> None:
    """The forward's plan, checked against the mask rather than against the arithmetic that made it.

    Three separate claims, and the middle one is the rung's:
      * blocks below ``full_end`` are wholly visible AND wholly in range, so the kernel's first
        loop may drop the causal mask and the key-bounds mask together;
      * blocks in ``[full_end, masked_end)`` are the only ones that need either;
      * blocks at or beyond ``masked_end`` contain no visible key at all and are never loaded.
    """
    vis = _visible(n_ctx, causal)
    for q_start in range(0, n_ctx, bq):
        span = kv_span(q_start=q_start, block_q=bq, block_k=bk, n_ctx=n_ctx, causal=causal)
        rows = vis[q_start : min(q_start + bq, n_ctx)]
        for k_start in range(0, ((n_ctx + bk - 1) // bk) * bk, bk):
            block = rows[:, k_start : min(k_start + bk, n_ctx)]
            in_range = k_start + bk <= n_ctx
            if k_start < span.full_end:
                assert in_range and bool(block.all()), (
                    f"unmasked block at k={k_start} for q={q_start} is not wholly visible/in-range"
                )
            elif k_start < span.masked_end:
                assert bool(block.any()), f"masked block at k={k_start} has nothing to contribute"
            else:
                assert block.numel() == 0 or not bool(block.any()), (
                    f"block at k={k_start} was skipped but contains a visible key"
                )


@pytest.mark.parametrize(("bq", "bk", "n_ctx", "causal"), _SPAN_CASES)
def test_q_span_visits_exactly_the_query_blocks_that_reach_this_kv_block(
    bq: int, bk: int, n_ctx: int, causal: bool
) -> None:
    """The dK/dV backward's plan — :func:`kv_span` with the roles exchanged."""
    vis = _visible(n_ctx, causal)
    for kv_start in range(0, n_ctx, bk):
        span = q_span(kv_start=kv_start, block_q=bq, block_k=bk, n_ctx=n_ctx, causal=causal)
        cols = vis[:, kv_start : min(kv_start + bk, n_ctx)]
        for q_start in range(0, ((n_ctx + bq - 1) // bq) * bq, bq):
            block = cols[q_start : min(q_start + bq, n_ctx)]
            if q_start < span.masked_start:
                assert block.numel() == 0 or not bool(block.any()), (
                    f"query block {q_start} was skipped but contributes to KV block {kv_start}"
                )
            elif q_start >= span.full_start and q_start + bq <= n_ctx:
                assert bool(block.all()), (
                    f"query block {q_start} was treated as unmasked but is not wholly visible"
                )


def test_causal_block_skipping_halves_the_visited_blocks() -> None:
    """The saving, stated as a number rather than as an intention.

    A dense pass over 512 keys in 64-wide blocks is 8 blocks per query tile, 64 in total. Causal
    attention has half the work, and a kernel that merely masks does all 64 anyway — which is what
    ``fa2.py`` does for the mask expression even though ``fa2.py:82`` already clamps its loop.
    """
    n_ctx, bq, bk = 512, 64, 64
    causal_blocks = sum(
        (kv_span(q_start=q, block_q=bq, block_k=bk, n_ctx=n_ctx, causal=True).masked_end + bk - 1)
        // bk
        for q in range(0, n_ctx, bq)
    )
    dense_blocks = (n_ctx // bq) * (n_ctx // bk)
    assert causal_blocks == 36  # 1 + 2 + ... + 8
    assert dense_blocks == 64
    masked = sum(
        kv_span(q_start=q, block_q=bq, block_k=bk, n_ctx=n_ctx, causal=True).masked_span
        for q in range(0, n_ctx, bq)
    )
    assert masked == 8 * bk, "exactly one diagonal block per query tile should carry a mask"


def test_bwd_grids_cover_every_block_exactly_once() -> None:
    """Each output block is owned by one program — the property that removes the atomics."""
    dkdv = FA2Config(64, 64, 4, 2)
    dq = FA2Config(128, 64, 8, 2)
    grids = bwd_grids(n_ctx=500, batch=6, dkdv=dkdv, dq=dq)
    assert grids.dkdv == (-(-500 // dkdv.block_k), 6)
    assert grids.dq == (-(-500 // dq.block_q), 6)
    assert grids.dkdv[0] * dkdv.block_k >= 500 and grids.dq[0] * dq.block_q >= 500


# =============================================================================================
# CPU — exp2 and the LSE units
# =============================================================================================


def test_log2e_prescaled_scale_is_the_same_function_as_exp() -> None:
    """``exp2(s * scale * log2e) == exp(s * scale)`` — the whole justification for prescaling.

    ``oss/vllm/vllm/v1/attention/ops/triton_prefill_attention.py:244`` folds the constant into the
    scale on the host for exactly this reason and then calls ``tl.math.exp2`` (``:175``, ``:179``).
    """
    torch.manual_seed(0)
    s = torch.randn(8, 16, dtype=torch.float64) * 4.0
    nat = softmax_scale(D, use_exp2=False)
    exp2 = softmax_scale(D, use_exp2=True)
    assert exp2 == pytest.approx(nat * LOG2E)
    assert torch.allclose(torch.exp2(s * exp2), torch.exp(s * nat), rtol=1e-12, atol=0.0)


def test_lse_leaves_in_natural_log_units_whichever_base_the_mainloop_ran_in() -> None:
    """The unit slip that a forward test cannot catch.

    ``O = acc / l_i`` never touches ``m_i``, so a forward that forgot the ``ln 2`` passes its own
    correctness gate and hands the backward an LSE that is wrong by a factor of
    ``exp((1 - ln2) * m)`` per row. Both bases are checked against ``torch.logsumexp`` of the
    natural-log scores, which is what the backward kernels and reference.py consume.
    """
    torch.manual_seed(1)
    s = torch.randn(4, 32, dtype=torch.float64) * 3.0
    want = torch.logsumexp(s, dim=-1)

    m_nat = s.amax(dim=-1)
    l_nat = torch.exp(s - m_nat[:, None]).sum(dim=-1)
    assert torch.allclose(lse_from_running_state(m_nat, l_nat, use_exp2=False), want, atol=1e-12)

    s2 = s * LOG2E
    m2 = s2.amax(dim=-1)
    l2 = torch.exp2(s2 - m2[:, None]).sum(dim=-1)
    assert torch.allclose(lse_from_running_state(m2, l2, use_exp2=True), want, atol=1e-12)
    assert pytest.approx(1.0 / LOG2E) == LN2


# =============================================================================================
# CPU — the wrapper's guard clauses and the source it drives
# =============================================================================================


def _t(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.zeros(*shape, dtype=dtype)


def test_unsupported_reason_names_every_shape_this_rung_cannot_address() -> None:
    """Each of these is a property of what was built, and each must be said rather than launched.

    GQA is the one worth naming twice: ``bench/kernels/attention/k2_ladder.py`` carries a
    ``gqa8k`` shape, this wrapper cannot express it (it flattens all leading dims into one batch
    axis, so a KV head cannot be shared), and the Rung row therefore must not list it. A rung that
    advertises a shape its wrapper rejects measures a ValueError.
    """
    ok = (_t(2, 8, 128, 64), _t(2, 8, 128, 64), _t(2, 8, 128, 64))
    assert unsupported_reason(*ok) is None
    assert "no GQA" in (
        unsupported_reason(_t(2, 8, 128, 64), _t(2, 1, 128, 64), _t(2, 1, 128, 64)) or ""
    )
    assert "self-attention" in (
        unsupported_reason(_t(2, 128, 64), _t(2, 96, 64), _t(2, 96, 64)) or ""
    )
    assert "bytes/element" in (
        unsupported_reason(*(_t(2, 128, 64, dtype=torch.float32),) * 3) or ""
    )
    assert "power of two" in (
        unsupported_reason(_t(2, 128, 96), _t(2, 128, 96), _t(2, 128, 96)) or ""
    )
    assert "share a dtype" in (
        unsupported_reason(_t(2, 128, 64), _t(2, 128, 64, dtype=torch.float16), _t(2, 128, 64))
        or ""
    )


def test_the_forward_kernel_calls_the_hole_from_both_phases() -> None:
    """One hole, two call sites — the unmasked loop and the diagonal loop.

    If a later edit inlined one of them, the "fewer O-rescales" lesson would be half-learned and
    the two loops could drift into two different rescale rules without anything noticing.
    """
    src = _tuned_code()
    body = src[src.index("def _fa2_tuned_fwd_kernel") : src.index("def _fa2_tuned_bwd_dkdv_kernel")]
    assert body.count("_online_softmax_step(") == 2, (
        "the forward must call the rescale core once per phase; found "
        f"{body.count('_online_softmax_step(')} call(s)"
    )


def test_neither_backward_kernel_uses_an_atomic() -> None:
    """The delta against ``fa2.py:293``, asserted rather than described.

    Splitting the backward is only worth doing if it removes the atomics; if one crept back the
    split would be pure overhead and the rung would be measuring nothing.
    """
    assert "atomic_add" not in _tuned_code()


def test_the_backward_separates_the_exponent_scale_from_the_gradient_scale() -> None:
    """Two scales, and confusing them scales every gradient by log2(e) = 1.4427.

    A wrong-by-a-constant gradient still trains, which is why this is checked in the source rather
    than left to a tolerance to notice.
    """
    src = _tuned_code()
    assert "scale_grad = softmax_scale(d, use_exp2=False)" in src
    assert "scale_exp = softmax_scale(d, use_exp2=use_exp2)" in src
    assert "* scale_grad" in src and "* scale_exp" in src


# =============================================================================================
# gpu — correctness against the oracle. Needs a CUDA device with Triton.
# =============================================================================================


def _cuda_or_skip():  # noqa: ANN202
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    pytest.importorskip("triton")
    from scratch_llm.kernels.attention.prefill import fa2_tuned

    return fa2_tuned


def _qkv(b: int, h: int, n: int, d: int, seed: int = 0):  # noqa: ANN202
    torch.manual_seed(seed)
    gen = (torch.randn(b, h, n, d, device="cuda", dtype=torch.float32) for _ in range(3))
    return tuple(t.bfloat16() for t in gen)


@pytest.mark.gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
@pytest.mark.parametrize("n_ctx", [512, 500], ids=["aligned", "ragged"])
def test_forward_matches_the_oracle(causal: bool, n_ctx: int) -> None:
    """The tiled FA2 oracle in ``kernels/attention/reference.py``, up-cast over the same bf16
    inputs — the same rule K1 uses: the oracle must not be a different problem."""
    mod = _cuda_or_skip()
    q, k, v = _qkv(2, 4, n_ctx, D)
    out, lse = mod.fa2_tuned_forward(q, k, v, is_causal=causal)
    ref, ref_lse = flash_attention_forward(q.float(), k.float(), v.float(), is_causal=causal)
    tol = _require_tolerance("TOL_FWD", TOL_FWD)
    assert (out.float() - ref).abs().max().item() <= tol
    assert (lse - ref_lse).abs().max().item() <= tol, (
        "LSE is out of tolerance while O is inside it — look at the exp2 epilogue first "
        "(m_i * ln2), because O never touches m_i and cannot see that bug"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_the_exp2_path_and_the_exp_path_agree(causal: bool) -> None:
    """The lever, isolated. These are the same function; if they disagree the prescale is wrong."""
    mod = _cuda_or_skip()
    q, k, v = _qkv(2, 4, 512, D, seed=3)
    o2, l2 = mod.fa2_tuned_forward(q, k, v, is_causal=causal, use_exp2=True)
    on, ln = mod.fa2_tuned_forward(q, k, v, is_causal=causal, use_exp2=False)
    tol = _require_tolerance("TOL_FWD", TOL_FWD)
    assert (o2.float() - on.float()).abs().max().item() <= tol
    assert (l2 - ln).abs().max().item() <= tol


@pytest.mark.gpu
def test_block_skipping_does_not_change_the_answer() -> None:
    """The second lever, isolated: skipping blocks is an optimisation, not an approximation."""
    mod = _cuda_or_skip()
    q, k, v = _qkv(2, 4, 500, D, seed=4)
    fast, _ = mod.fa2_tuned_forward(q, k, v, is_causal=True, skip_masked_blocks=True)
    slow, _ = mod.fa2_tuned_forward(q, k, v, is_causal=True, skip_masked_blocks=False)
    tol = _require_tolerance("TOL_FWD", TOL_FWD)
    assert (fast.float() - slow.float()).abs().max().item() <= tol


@pytest.mark.gpu
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "dense"])
def test_backward_matches_the_pytorch_oracle(causal: bool) -> None:
    """dQ, dK, dV against ``FlashAttentionPyTorch``'s recomputation backward.

    The oracle is autograd over the same recurrence, so a mismatch is this rung's two kernels, not
    a different definition of the gradient.
    """
    mod = _cuda_or_skip()
    q, k, v = _qkv(2, 4, 512, D, seed=5)
    qr, kr, vr = (t.float().detach().requires_grad_(True) for t in (q, k, v))
    o_ref = FlashAttentionPyTorch.apply(qr, kr, vr, causal)
    torch.manual_seed(6)
    # Rounded to bf16 ONCE and then shared. Handing the oracle the fp32 dO and the kernel its bf16
    # rounding would make them different problems, and the rounding error would sit inside the
    # tolerance where a real bug could hide under it — the same rule as the forward's oracle.
    do = torch.randn_like(o_ref).bfloat16()
    o_ref.backward(do.float())

    o, lse = mod.fa2_tuned_forward(q, k, v, is_causal=causal)
    dq, dk, dv = mod.fa2_tuned_backward(q, k, v, o, lse, do, is_causal=causal)
    tol = _require_tolerance("TOL_BWD", TOL_BWD)
    for got, want, name in (
        (dq, qr.grad, "dQ"),
        (dk, kr.grad, "dK"),
        (dv, vr.grad, "dV"),
    ):
        assert want is not None, f"the fp32 oracle produced no {name} gradient to compare against"
        err = (got.float() - want).abs().max().item()
        assert err <= tol, (
            f"max|Δ{name}| = {err:.4e} > {tol:.4e}. dV out alone points at the P cast; dQ and dK "
            f"out together with dV inside points at the scale_exp/scale_grad mix-up, whose "
            f"signature is a uniform factor of {LOG2E:.4f}."
        )


@pytest.mark.gpu
def test_tuned_forward_agrees_with_the_untuned_kernel_it_replaces() -> None:
    """A-R1's number is a delta on ``fa2.py``; the two must compute the same thing to be compared."""
    mod = _cuda_or_skip()
    q, k, v = _qkv(2, 4, 512, D, seed=7)
    tuned, _ = mod.fa2_tuned_forward(q, k, v, is_causal=True)
    base, _ = mod.baseline_forward(q, k, v, is_causal=True)
    tol = _require_tolerance("TOL_FWD", TOL_FWD)
    assert (tuned.float() - base.float()).abs().max().item() <= tol


@pytest.mark.gpu
@pytest.mark.parametrize("kind", KINDS)
def test_this_device_actually_has_configs(kind: Kind) -> None:
    """The arch filter, run against the card in the box rather than against the table.

    ``arch_for`` raises for a compute capability with no row, which is the intended failure — but
    it should be met here, in a named test, and not inside an autotune on a rented hour.
    """
    _cuda_or_skip()
    from scratch_llm.kernels.common.arch import compute_capability

    cc = compute_capability()
    assert cc is not None, "CUDA is available but reports no compute capability"
    chosen = require_configs(cc, D, kind=kind)
    assert chosen and all(
        smem_bytes(c, D, kind=kind) <= ARCH[f"sm_{cc[0] * 10 + cc[1]}a"].smem_per_cta
        for c in chosen
    )


@pytest.mark.gpu
def test_rejects_the_shapes_it_cannot_address() -> None:
    """The guard clauses, on the device where they can be reached."""
    _cuda_or_skip()
    from scratch_llm.kernels.attention.prefill.fa2_tuned import fa2_tuned_forward

    def z(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(*shape, device="cuda", dtype=dtype)

    with pytest.raises(ValueError, match="no GQA"):
        fa2_tuned_forward(z(2, 8, 128, 64), z(2, 1, 128, 64), z(2, 1, 128, 64))
    with pytest.raises(ValueError, match="self-attention"):
        fa2_tuned_forward(z(2, 128, 64), z(2, 96, 64), z(2, 96, 64))
    with pytest.raises(ValueError, match="bytes/element"):
        fa2_tuned_forward(*(z(2, 128, 64, dtype=torch.float32),) * 3)
