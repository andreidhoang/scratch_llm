"""S1/S-R4 — fused MoE. Two tiers of gate: everything the CPU can decide, then the box.

The CPU tier carries almost the whole weight of this rung, for a reason specific to MoE: every
index bug in a permuted MoE produces a plausible number. An inverse permutation that is off by the
top_k stride returns each token's output to a neighbouring token — the layer still runs, the loss
still decreases, and the model is quietly worse. A group offset that skips an empty expert shifts
every later expert's rows by one, so 127 of 128 experts multiply somebody else's tokens. A tie
broken differently on two devices changes which experts a prompt visits. None of those crash, none
of them are slow, and all of them are decidable on a laptop in milliseconds.

So the CPU tier tests the four things the GEMM's correctness rests on — the tie, the permutation
and its exact inverse, the group offsets (with an empty group, deliberately), and the load
accounting — plus the whole pipeline end to end against an oracle that shares no code with it, by
injecting the torch grouped GEMM in place of the Triton one.

The GPU tier is the Triton kernel against the fp32 oracle, and the layer against vLLM's
``fused_experts`` at the identical routing. It needs an H100 and says so.

Spec: experiments/S1/S-R4/spec.md   ·   Map: experiments/S1/S-R4/map.md
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from scratch_llm.kernels.moe.ep import (
    ExpertParallel,
    comm_fraction,
    counts_by_expert,
    plan_all_ranks,
    plan_dispatch,
    plan_receive,
    traffic,
)
from scratch_llm.kernels.moe.fused import (
    reference_moe,
    s_r4_fused_moe,
    silu_and_mul,
    torch_grouped_gemm,
    trace_moe,
)
from scratch_llm.kernels.moe.grouped_gemm import (
    default_config,
    exact_m_tiles,
    max_m_tiles,
    reference_grouped_gemm,
    validate_group_offsets,
)
from scratch_llm.kernels.moe.routing import (
    Routing,
    build_permutation,
    combine,
    expert_load,
    gather_tokens,
    permute_rows,
    route_topk,
    topk_deterministic,
    unpermute_rows,
)

_REPO = Path(__file__).resolve().parents[3]
SOURCE = "src/scratch_llm/kernels/moe/grouped_gemm_triton.py"
RUNG = "S1/S-R4"

#: The box command every GPU gate prints instead of passing quietly.
_NO_GPU = (
    "no CUDA device — this gate is an H100 statement. On the box:  infra/rent.sh sync-up "
    "<user@host> && ssh in && cd scratch_llm && pytest tests/kernels/moe/test_s1_s_r4.py -m gpu"
)

# ---------------------------------------------------------------------------------------------
# The tolerance. HUY SETS THIS, with its argument, before the first measured run.
#
# The gate's form is  max|y_kernel − y_fp32| / max|y_fp32|  <=  TOL_REL, on a bf16 layer output.
# Three error sources have to be separated before a constant can be defended: the bf16 input
# quantum (2^-8 relative) carried through two GEMMs of K=2048 and K=768; the fp32 accumulator's
# own √K random walk; and the reduction over top_k=8 gated contributions, which this design does
# in fp32 and vLLM's `moe_sum` does in bf16 — so the rung's output and the floor's differ by an
# amount that is NOT the kernel being wrong, and the constant has to be wide enough to say so
# while still being narrow enough to catch a permutation that is off by one row.
#
# Leaving it None is deliberate: choosing it is the rung's numerics lesson, and the argument
# belongs in spec.md's "Correctness gate" line.
# ---------------------------------------------------------------------------------------------
TOL_REL: float | None = None

#: The plan's row, verbatim (SIXTY_DAYS_SIX_LADDERS.md §05, S-R4). Small enough to test at.
SPEC = {"n_experts": 128, "top_k": 8, "hidden": 2048, "intermediate": 768}

#: A tiny model the CPU tier runs at. Same structure, four orders of magnitude cheaper.
TINY = {"n_experts": 6, "top_k": 2, "hidden": 8, "intermediate": 4}


def _hole_is_open() -> bool:
    """True while the mainloop still carries the sentinel (same rule as tests/conftest.py).

    A regex, not a substring: ``ruff format`` may wrap the raise's argument onto its own line, and
    a substring check would then report the hole CLOSED — flipping the guard test green over an
    unwritten kernel, which is the one failure this whole convention exists to prevent.
    """
    text = (_REPO / SOURCE).read_text(encoding="utf-8")
    return re.search(r'NotImplementedError\(\s*(?:#[^\n]*\n\s*)?"HUY:', text) is not None


def _routing(topk_ids: list[list[int]], n_experts: int, weight: float | None = None) -> Routing:
    """A hand-built routing — the point of the CPU tier is routings randomness never produces."""
    ids = torch.tensor(topk_ids, dtype=torch.int32)
    w = torch.full(ids.shape, 1.0 / ids.shape[1] if weight is None else weight, dtype=torch.float32)
    return Routing(ids, w, n_experts=n_experts)


def _tiny_weights(seed: int = 0, **shape) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN003
    g = torch.Generator().manual_seed(seed)
    e, i, h = shape["n_experts"], shape["intermediate"], shape["hidden"]
    w1 = torch.randn(e, 2 * i, h, generator=g) * h**-0.5
    w2 = torch.randn(e, h, i, generator=g) * i**-0.5
    return w1, w2


# =============================================================================================
# The hole guard — exactly one (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("S1/S-R4", "src/scratch_llm/kernels/moe/grouped_gemm_triton.py")
def test_grouped_gemm_mainloop_is_filled_and_the_module_imports() -> None:
    """Fails while ``_grouped_gemm_kernel`` raises; passes once the mainloop is written.

    A Triton rung has no compile step that runs on a laptop — ``@triton.jit`` parses the body at
    first compile, on the device — so this guard asserts the only two things that need no silicon:
    that the body exists, and, where Triton is installed, that the module still imports with it.
    That second half is weak on purpose. Correctness is
    ``test_triton_grouped_gemm_matches_the_fp32_oracle``, on the box, at Huy's tolerance.
    """
    assert not _hole_is_open(), (
        f"{RUNG} grouped-GEMM mainloop is still an open hole. Read experiments/S1/S-R4/spec.md and "
        f"map.md, then write the per-group tile mapping and the accumulate/epilogue in {SOURCE}. "
        f"Everything around it — routing, permutation, group offsets, un-permute, the launcher, "
        f"the grid, the EP plans, the oracle and every test below — is already there."
    )
    pytest.importorskip("triton", reason="Triton absent here; the import check is box-only")
    from scratch_llm.kernels.moe.grouped_gemm_triton import grouped_gemm_triton

    assert grouped_gemm_triton is not None


def test_the_hole_sentinel_survives_the_formatter() -> None:
    """``ruff format`` has broken this convention before, and it fails SILENTLY.

    ``tests/conftest.py`` decides a hole is open by matching a sentinel in the source. If the
    formatter wraps the raise's message onto the next line the sentinel disappears, the strict
    xfail is never applied, and the guard above goes green over an unwritten kernel. So: the
    source must still match the conftest rule, and the marker comment must still name this rung.
    """
    text = (_REPO / SOURCE).read_text(encoding="utf-8")
    if not _hole_is_open():
        # Filled. The marker comment is SUPPOSED to be gone — infra/holes.py is the ledger of what
        # is left, so a filled hole drops out of it. All that is left to check is that the source
        # is not still raising behind a sentinel the conftest rule can no longer see.
        assert "raise NotImplementedError" not in text, (
            f"{SOURCE} raises NotImplementedError but conftest's sentinel no longer matches it — "
            f"the hole is invisible to the inventory and its guard test reports green while unwritten."
        )
        return
    marker = re.search(r"HUY:.*?spec:\s*experiments/(\S+?)/spec\.md", text)
    assert marker is not None, f"{SOURCE} has no `# HUY:` marker for infra/holes.py to find"
    assert marker.group(1) == RUNG, f"the marker points at {marker.group(1)}, not {RUNG}"


# =============================================================================================
# CPU — the tie
# =============================================================================================


def test_topk_ties_break_toward_the_lower_expert_index() -> None:
    """All-equal scores must select 0..k-1, on every device, every time.

    ``torch.topk`` does not promise this and its answer is backend-dependent. A router that picks
    differently on CPU and GPU makes every cross-device comparison in this rung — including the
    one against vLLM — a comparison of two different models.
    """
    scores = torch.full((3, 8), 0.125)
    _, ids = topk_deterministic(scores, 4)
    assert ids.tolist() == [[0, 1, 2, 3]] * 3


def test_topk_breaks_a_partial_tie_at_the_boundary() -> None:
    """The tie that matters is at the k-th place, where two experts contend for the last slot."""
    scores = torch.tensor([[0.9, 0.1, 0.5, 0.5, 0.5]])
    values, ids = topk_deterministic(scores, 3)
    assert ids.tolist() == [[0, 2, 3]], "the last slot must go to the lowest-indexed contender"
    torch.testing.assert_close(values, torch.tensor([[0.9, 0.5, 0.5]]))


def test_topk_agrees_with_torch_topk_when_nothing_is_tied() -> None:
    """Determinism must not have cost correctness: with distinct scores, the two agree exactly."""
    torch.manual_seed(0)
    scores = torch.rand(16, 32) + torch.arange(32) * 1e-3
    values, ids = topk_deterministic(scores, 5)
    ref = torch.topk(scores, 5, dim=-1)
    assert torch.equal(ids.to(torch.int64), ref.indices)
    assert torch.allclose(values, ref.values)


def test_route_topk_renormalizes_over_the_selected_experts() -> None:
    """Qwen3's ``norm_topk_prob``: the k gates sum to 1, and they are fp32 whatever the logits are."""
    torch.manual_seed(0)
    routing = route_topk(torch.randn(9, 16, dtype=torch.bfloat16).float(), 4)
    assert routing.validate() == []
    assert routing.topk_weights.dtype is torch.float32
    torch.testing.assert_close(routing.topk_weights.sum(-1), torch.ones(9), rtol=1e-6, atol=1e-6)


# =============================================================================================
# CPU — the permutation and its exact inverse
# =============================================================================================


def _degenerate_routings() -> list[tuple[str, Routing]]:
    """The routings a uniform random test never produces, and a real decode step does."""
    torch.manual_seed(0)
    n_experts, top_k, n_tokens = 6, 2, 9
    random_ids = torch.stack([torch.randperm(n_experts)[:top_k] for _ in range(n_tokens)]).to(
        torch.int32
    )
    return [
        ("random", Routing(random_ids, torch.rand(n_tokens, top_k), n_experts)),
        # Expert 3 gets nothing and experts 4, 5 get nothing either — the decode case.
        ("empty-experts", _routing([[0, 1], [0, 2], [1, 2], [0, 1]], n_experts)),
        # One expert takes every slot: group_offsets is [0,0,...,n_rows] with one jump.
        ("one-expert-takes-all", _routing([[2, 2]] * 5, n_experts)),
        # Expert 0 empty — the off-by-one that a `cumsum` without the leading zero produces.
        ("first-expert-empty", _routing([[1, 2], [3, 4], [5, 1]], n_experts)),
        # Last expert empty — the mirror image, which a `cumsum[:-1]` produces.
        ("last-expert-empty", _routing([[0, 1], [0, 2], [3, 4]], n_experts)),
        ("single-token", _routing([[4, 5]], n_experts)),
    ]


@pytest.mark.parametrize(
    "name,routing", _degenerate_routings(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_permute_then_unpermute_is_the_identity(name: str, routing: Routing) -> None:
    """``unpermute(permute(x)) == x``, exactly, for every routing including the degenerate ones.

    The payload is an integer ramp so the comparison is exact: a floating payload would let a
    permutation that is wrong by a row of equal values pass.
    """
    p = build_permutation(routing.topk_ids, routing.n_experts)
    assert p.validate() == [], f"{name}: {p.validate()}"
    rows = torch.arange(p.n_rows, dtype=torch.float64).unsqueeze(1).repeat(1, 3)
    rows[:, 1] *= -1
    assert torch.equal(unpermute_rows(permute_rows(rows, p), p), rows), name
    # ... and the other way round, which is a different statement about the same pair.
    assert torch.equal(permute_rows(unpermute_rows(rows, p), p), rows), name


@pytest.mark.parametrize(
    "name,routing", _degenerate_routings(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_permuted_rows_are_expert_major_and_token_ordered_within_an_expert(
    name: str, routing: Routing
) -> None:
    """The two properties the grouped GEMM's group bounds depend on.

    Expert-major: row ``p`` belongs to expert ``e`` iff it lies in ``[offsets[e], offsets[e+1])``.
    Token-ordered within an expert: the stable sort keeps rows in increasing flat-slot order, which
    is what makes a row-for-row diff against vLLM (whose aligner also sorts stably) readable.
    """
    p = build_permutation(routing.topk_ids, routing.n_experts)
    flat = routing.topk_ids.reshape(-1).to(torch.int64)
    for e in range(routing.n_experts):
        start, stop = p.group_bounds(e)
        rows = p.perm[start:stop]
        assert torch.equal(flat[rows], torch.full((stop - start,), e, dtype=torch.int64)), (
            f"{name}: expert {e}"
        )
        assert torch.equal(rows, rows.sort().values), f"{name}: expert {e} is not token-ordered"


@pytest.mark.parametrize(
    "name,routing", _degenerate_routings(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_gather_tokens_puts_the_right_token_in_every_row(name: str, routing: Routing) -> None:
    """The expansion half of the permutation: row ``p`` must carry token ``perm[p] // top_k``."""
    p = build_permutation(routing.topk_ids, routing.n_experts)
    x = torch.arange(p.n_tokens, dtype=torch.float64).unsqueeze(1)
    rows = gather_tokens(x, p)
    assert torch.equal(rows.squeeze(1), (p.perm // routing.top_k).double()), name


def test_combine_applies_the_gate_and_reduces_over_top_k() -> None:
    """Un-permute, scale by the gate, sum the k contributions — in that order.

    Built so the right answer is known without recomputing the permutation: every expert output is
    1, so each token's result is the sum of its own gates.
    """
    routing = _routing([[0, 3], [1, 1], [2, 0]], n_experts=4)
    routing = Routing(routing.topk_ids, torch.tensor([[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]]), 4)
    p = build_permutation(routing.topk_ids, 4)
    ones = torch.ones(p.n_rows, 2)
    out = combine(ones, p, routing.topk_weights)
    torch.testing.assert_close(out, torch.ones(3, 2), rtol=0, atol=1e-6)


def test_combine_routes_each_expert_output_back_to_its_own_token() -> None:
    """The bug this catches: an inverse that is off by the top_k stride. Every token gets a
    distinct expert output, so a shifted inverse lands the wrong value on the wrong token."""
    routing = _routing([[0], [1], [2], [3]], n_experts=4, weight=1.0)
    p = build_permutation(routing.topk_ids, 4)
    # Permuted-row payload = the expert id it belongs to; token t chose expert t, so out[t] == t.
    payload = p.perm.double().unsqueeze(1) * 0 + torch.tensor(
        [[float(routing.topk_ids.reshape(-1)[s])] for s in p.perm.tolist()], dtype=torch.float64
    )
    out = combine(payload, p, routing.topk_weights.double())
    assert out.squeeze(1).tolist() == [0.0, 1.0, 2.0, 3.0]


# =============================================================================================
# CPU — the group offsets. The empty group is the case, not the edge case.
# =============================================================================================


@pytest.mark.parametrize(
    "name,routing", _degenerate_routings(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_group_offsets_are_monotone_start_at_zero_and_total_the_rows(
    name: str, routing: Routing
) -> None:
    p = build_permutation(routing.topk_ids, routing.n_experts)
    off = p.group_offsets
    assert off.numel() == routing.n_experts + 1, name
    assert int(off[0]) == 0 and int(off[-1]) == p.n_rows == routing.n_tokens * routing.top_k, name
    assert bool((off[1:] >= off[:-1]).all()), f"{name}: {off.tolist()} is not monotone"
    assert validate_group_offsets(off, n_rows=p.n_rows, n_experts=routing.n_experts) == [], name


def test_an_empty_expert_is_a_zero_width_group_and_shifts_nothing() -> None:
    """THE case that appears at decode B=64 and never in a uniform random test.

    Hand-built so the expected offsets are known by hand: experts 1 and 4 get nothing.
    ``[0, 1], [0, 2], [2, 3], [0, 5]`` over 6 experts is 8 slots — expert 0 gets 3, expert 1 gets
    0, expert 2 gets 2, expert 3 gets 1, expert 4 gets 0, expert 5 gets 1, and every group after an
    empty one must still start where the previous one ended. An implementation that "skips" an
    empty expert shifts all of them by one and silently multiplies 5 experts' tokens by the wrong
    weights; one that emits a one-wide group for it drops a real token off the end.
    """
    routing = _routing([[0, 1], [0, 2], [2, 3], [0, 5]], n_experts=6)
    p = build_permutation(routing.topk_ids, 6)
    assert p.counts.tolist() == [3, 1, 2, 1, 0, 1]
    assert p.group_offsets.tolist() == [0, 3, 4, 6, 7, 7, 8]
    assert p.group_bounds(4) == (7, 7), (
        "an empty expert must be a zero-width group, not a missing one"
    )
    assert p.group_bounds(5) == (7, 8), "the group after an empty one must not be shifted"
    assert p.validate() == []
    assert int(p.group_offsets[-1]) == 8, "no slot may be lost to an empty group"


def test_an_expert_that_takes_everything_leaves_every_other_group_empty() -> None:
    routing = _routing([[3, 3]] * 4, n_experts=5)
    p = build_permutation(routing.topk_ids, 5)
    assert p.group_offsets.tolist() == [0, 0, 0, 0, 8, 8]
    assert p.group_bounds(3) == (0, 8)
    assert p.validate() == []


def test_group_offsets_reject_a_hand_broken_array() -> None:
    """The validator has to fail on the three ways an offsets array goes wrong, or it is decoration."""
    off = torch.tensor([0, 3, 2, 8], dtype=torch.int32)
    assert any("monotone" in m for m in validate_group_offsets(off, n_rows=8, n_experts=3))
    assert any(
        "must be 0" in m
        for m in validate_group_offsets(
            torch.tensor([1, 4, 8], dtype=torch.int32), n_rows=8, n_experts=2
        )
    )
    assert any(
        "n_rows" in m
        for m in validate_group_offsets(
            torch.tensor([0, 4, 7], dtype=torch.int32), n_rows=8, n_experts=2
        )
    )


# =============================================================================================
# CPU — the tile counts the launcher's grid is sized from
# =============================================================================================


@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
@pytest.mark.parametrize(
    "name,routing", _degenerate_routings(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_the_tile_bound_is_never_below_the_exact_tile_count(
    name: str, routing: Routing, block_m: int
) -> None:
    """The launcher launches ``max_m_tiles`` CTAs. If that is ever less than the exact count, tiles
    at the end are never launched and their rows are never written — silently."""
    p = build_permutation(routing.topk_ids, routing.n_experts)
    exact = exact_m_tiles(p.group_offsets, block_m)
    bound = max_m_tiles(p.n_rows, routing.n_experts, block_m)
    assert bound >= exact, f"{name} at BLOCK_M={block_m}: bound {bound} < exact {exact}"


def test_an_empty_group_costs_zero_tiles_and_a_single_row_costs_one() -> None:
    """``cdiv(0, bm) == 0`` and ``cdiv(1, bm) == 1``. Getting the second wrong drops a token."""
    off = torch.tensor([0, 0, 1, 1, 33], dtype=torch.int32)  # groups of 0, 1, 0, 32
    assert exact_m_tiles(off, 32) == 0 + 1 + 0 + 1
    assert exact_m_tiles(off, 16) == 0 + 1 + 0 + 2


def test_the_pipeline_keys_its_tile_shape_on_tokens_not_on_rows() -> None:
    """vLLM computes ONE config from ``M = num_tokens`` and uses it for both GEMMs
    (fused_moe.py:1737). Ours must too, or the comparison is not tile-for-tile: at decode the row
    count is 8x the token count and lands in a different branch (BLOCK_M 64 instead of 32)."""
    seen: list[object] = []

    def spy(a, b, group_offsets, *, config=None):  # noqa: ANN001, ANN202
        seen.append(config)
        return torch_grouped_gemm(a, b, group_offsets)

    routing = route_topk(torch.randn(64, TINY["n_experts"]), TINY["top_k"])
    w1, w2 = _tiny_weights(**TINY)
    s_r4_fused_moe(torch.randn(64, TINY["hidden"]), w1, w2, routing, gemm=spy)
    assert len(seen) == 2 and seen[0] is seen[1], "both GEMMs must get the same config object"
    assert seen[0] == default_config(
        n_tokens=64,
        n_experts=TINY["n_experts"],
        n_out=2 * TINY["intermediate"],
        k_dim=TINY["hidden"],
    )


def test_default_config_follows_vllms_batch_size_branches() -> None:
    """The floor picks its tiles by token count, not row count (fused_moe.py:1299 keyed on M =
    num_tokens). Matching those branches is what makes the comparison tile-for-tile."""
    assert default_config(n_tokens=64, n_experts=128, n_out=1536, k_dim=2048).block_m == 32
    assert default_config(n_tokens=4096, n_experts=128, n_out=1536, k_dim=2048).block_m == 128
    assert default_config(n_tokens=16, n_experts=128, n_out=1536, k_dim=2048).block_m == 16
    assert default_config(n_tokens=64, n_experts=128, n_out=1536, k_dim=2048).validate() == []


# =============================================================================================
# CPU — the load accounting
# =============================================================================================


@pytest.mark.parametrize(
    "name,routing", _degenerate_routings(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_expert_load_sums_to_tokens_times_top_k(name: str, routing: Routing) -> None:
    """The one invariant that catches a slot dropped anywhere upstream, and it costs nothing."""
    load = expert_load(routing.topk_ids, routing.n_experts)
    assert load.validate() == [], name
    assert load.total_slots == routing.n_tokens * routing.top_k, name
    assert int(load.counts.sum()) == load.expected_slots, name


def test_expert_load_counts_empty_experts_and_reports_the_imbalance() -> None:
    routing = _routing([[0, 1], [0, 2], [2, 3], [0, 5]], n_experts=6)
    load = expert_load(routing.topk_ids, 6)
    assert load.counts.tolist() == [3, 1, 2, 1, 0, 1]
    assert load.n_empty == 1
    assert load.max_over_mean == pytest.approx(3 * 6 / 8)
    row = load.as_row()
    assert row["n_empty_experts"] == 1 and row["max_slots"] == 3 and row["min_slots"] == 0


def test_load_accounting_matches_the_permutations_own_counts() -> None:
    """Two independent counts of the same thing — a disagreement means one of them is a bug."""
    torch.manual_seed(0)
    ids = torch.randint(0, 12, (37, 4), dtype=torch.int32)
    assert torch.equal(expert_load(ids, 12).counts, build_permutation(ids, 12).counts)


# =============================================================================================
# CPU — the grouped GEMM's oracle, and the whole pipeline against an oracle that shares no code
# =============================================================================================


def test_reference_grouped_gemm_multiplies_each_group_by_its_own_expert() -> None:
    """Including an empty group in the middle, which must consume no rows and shift nothing."""
    a = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    b = torch.stack([torch.eye(3), torch.eye(3) * 2, torch.eye(3) * 3])
    offsets = torch.tensor([0, 2, 2, 4], dtype=torch.int32)  # expert 1 is empty
    out = reference_grouped_gemm(a, b, offsets)
    torch.testing.assert_close(out[:2], a[:2] * 1.0)
    torch.testing.assert_close(out[2:], a[2:] * 3.0)


def test_grouped_gemm_refuses_offsets_that_do_not_cover_the_rows() -> None:
    a = torch.zeros(4, 3)
    b = torch.zeros(2, 3, 3)
    with pytest.raises(ValueError, match="group_offsets"):
        reference_grouped_gemm(a, b, torch.tensor([0, 2, 3], dtype=torch.int32))


def test_silu_and_mul_splits_gate_from_up() -> None:
    h = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    torch.testing.assert_close(silu_and_mul(h), torch.nn.functional.silu(h[:, :2]) * h[:, 2:])
    with pytest.raises(ValueError, match="2I"):
        silu_and_mul(torch.zeros(1, 3))


@pytest.mark.parametrize(
    "name,routing", _degenerate_routings(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_the_whole_pipeline_matches_the_oracle_on_cpu(name: str, routing: Routing) -> None:
    """route → permute → grouped GEMM → silu_and_mul → grouped GEMM → un-permute, against a loop
    that does none of those things.

    This is the test the injectable GEMM exists for. The oracle never builds a permutation, never
    computes an offset and never tiles, so agreement here is evidence about the index math and not
    about two copies of the same mistake.
    """
    n_experts = routing.n_experts
    w1, w2 = _tiny_weights(
        n_experts=n_experts, intermediate=TINY["intermediate"], hidden=TINY["hidden"]
    )
    torch.manual_seed(1)
    x = torch.randn(routing.n_tokens, TINY["hidden"])
    got = s_r4_fused_moe(x, w1, w2, routing, gemm=torch_grouped_gemm)
    want = reference_moe(x, w1, w2, routing)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6, msg=lambda m: f"{name}: {m}")


def test_the_pipeline_at_a_decode_shaped_routing_with_many_empty_experts() -> None:
    """64 tokens, top_k 8, 128 experts — the plan's decode point, at a toy hidden size.

    The assertion that matters is not the number, it is that some experts got nothing: this is the
    shape where the empty-group path is exercised by the routing itself rather than by hand.
    """
    torch.manual_seed(0)
    n_experts, top_k = SPEC["n_experts"], SPEC["top_k"]
    routing = route_topk(torch.randn(64, n_experts), top_k)
    load = expert_load(routing.topk_ids, n_experts)
    assert load.n_empty > 0, (
        "512 slots over 128 experts should leave some empty — the routing changed"
    )
    assert load.total_slots == 64 * top_k
    w1, w2 = _tiny_weights(n_experts=n_experts, intermediate=4, hidden=8)
    x = torch.randn(64, 8)
    got = s_r4_fused_moe(x, w1, w2, routing, gemm=torch_grouped_gemm)
    want = reference_moe(x, w1, w2, routing)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_trace_reports_the_tile_padding_the_decode_point_pays() -> None:
    """The diagnostic the bench records: at 512 rows over 128 experts, the m-tiles are mostly air."""
    torch.manual_seed(0)
    routing = route_topk(torch.randn(64, SPEC["n_experts"]), SPEC["top_k"])
    trace = trace_moe(routing, hidden=SPEC["hidden"], intermediate=SPEC["intermediate"])
    assert set(trace.as_row()) >= {"exact_m_tiles", "max_m_tiles", "load", "config"}
    block_m = trace.config.block_m
    assert exact_m_tiles(trace.permutation.group_offsets, block_m) <= max_m_tiles(
        trace.n_rows, SPEC["n_experts"], block_m
    )
    assert trace.row_padding_ratio > 4.0, "512 rows over ~126 tiles of 32 should be >4x padded"
    assert trace.flops == 6.0 * 512 * SPEC["hidden"] * SPEC["intermediate"]


# =============================================================================================
# CPU — expert parallelism: placement, dispatch, and the receiver's regroup
# =============================================================================================


def test_expert_placement_is_contiguous_and_matches_vllms_linear_rule() -> None:
    """vLLM's ``determine_expert_map`` with ``linear`` placement: contiguous chunks, the remainder
    to the LOW ranks (expert_map_manager.py:69-77). Ours has to mean the same thing or an expert id
    means two different things on the two sides of the comparison."""
    ep = ExpertParallel(n_experts=128, ep_size=2)
    assert ep.local_range(0) == (0, 64) and ep.local_range(1) == (64, 128)
    assert ep.expert_map(1)[64].item() == 0 and ep.expert_map(1)[63].item() == -1
    uneven = ExpertParallel(n_experts=7, ep_size=2)
    assert uneven.local_range(0) == (0, 4) and uneven.local_range(1) == (4, 7)
    assert [uneven.owner_of(e) for e in range(7)] == [0, 0, 0, 0, 1, 1, 1]


def test_dispatch_counts_account_for_every_slot() -> None:
    torch.manual_seed(0)
    ep = ExpertParallel(n_experts=8, ep_size=2)
    ids = route_topk(torch.randn(5, 8), 3).topk_ids
    plan = plan_dispatch(ids, ep, rank=0)
    assert plan.send_total == 5 * 3
    assert int(plan.send_counts.sum()) == int(counts_by_expert(ids, ep).sum())
    assert plan.send_to_peers == plan.send_total - int(plan.send_counts[0])


def test_receive_counts_are_the_transpose_of_the_send_counts() -> None:
    """Across the group nothing is created or lost: what rank s sends to d is what d receives."""
    torch.manual_seed(0)
    ep = ExpertParallel(n_experts=8, ep_size=2)
    routings = [route_topk(torch.randn(6, 8), 2).topk_ids for _ in range(2)]
    plans = plan_all_ranks(routings, ep)
    assert all(p.validate() == [] for p in plans)
    for dst, p in enumerate(plans):
        for src, q in enumerate(plans):
            assert int(p.recv_counts[src]) == int(q.send_counts[dst])
    assert sum(p.recv_total for p in plans) == sum(p.send_total for p in plans)


def test_receive_plan_regroups_source_major_arrivals_into_expert_groups() -> None:
    """Hand-checked, including an empty (source, expert) block — the merge of sorted runs.

    Rank 0 sends 2 rows for local expert 0 and 3 for local expert 2 (nothing for 1); rank 1 sends
    1 for expert 0 and 4 for expert 1. Arrivals are source-major; the GEMM needs expert-major.
    """
    counts = torch.tensor([[2, 0, 3], [1, 4, 0]])
    plan = plan_receive(counts)
    assert plan.validate() == []
    assert plan.group_offsets.tolist() == [0, 3, 7, 10]
    assert plan.regroup.tolist() == [0, 1, 5, 6, 7, 8, 9, 2, 3, 4]
    rows = torch.arange(10, dtype=torch.float64).unsqueeze(1)
    assert torch.equal(plan.undo(plan.apply(rows)), rows), "the combine must be the exact transpose"


def test_receive_plan_survives_a_rank_that_sends_nothing() -> None:
    plan = plan_receive(torch.tensor([[0, 0], [3, 2]]))
    assert plan.validate() == []
    assert plan.group_offsets.tolist() == [0, 3, 5]
    assert plan.regroup.tolist() == [0, 1, 2, 3, 4]


def test_comm_fraction_is_the_ratio_of_two_measured_times() -> None:
    """The plan row's number. Deliberately not derived from bytes and a link speed: at decode the
    all-to-all is latency-bound, which is where a bandwidth estimate would be most wrong and most
    convincing. A layer time of zero is not a fraction and must raise, not return inf."""
    assert comm_fraction(0.25, 1.0) == pytest.approx(0.25)
    assert comm_fraction(1.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="layer_ms"):
        comm_fraction(0.5, 0.0)


def test_traffic_counts_only_what_crosses_the_wire() -> None:
    """Rows a rank sends to itself never touch NVLink; counting them would flatter the link."""
    torch.manual_seed(0)
    ep = ExpertParallel(n_experts=8, ep_size=2)
    plans = plan_all_ranks([route_topk(torch.randn(8, 8), 4).topk_ids for _ in range(2)], ep)
    tr = traffic(plans, hidden=2048, elem_bytes=2)
    assert tr.dispatch_rows_total == 2 * 8 * 4
    assert tr.dispatch_rows_on_wire == sum(p.send_to_peers for p in plans)
    assert 0.0 <= tr.peer_fraction <= 1.0
    assert tr.combine_bytes_on_wire == tr.dispatch_bytes_on_wire
    assert tr.total_bytes_on_wire == 2 * tr.dispatch_rows_on_wire * 2048 * 2


# =============================================================================================
# gpu — the Triton kernel against the oracle, and the layer against vLLM. H100.
# =============================================================================================


def _require_tolerance() -> float:
    if TOL_REL is None:
        pytest.fail(
            "TOL_REL is unset. The correctness gate cannot run without a tolerance, and picking one "
            "is this rung's numerics lesson: two bf16 GEMMs with fp32 accumulators, then a top_k=8 "
            "reduction this design does in fp32 and vLLM's moe_sum does in bf16. Set it in "
            "tests/kernels/moe/test_s1_s_r4.py and write the one-line argument into "
            "experiments/S1/S-R4/spec.md's 'Correctness gate' line before the first measured run."
        )
    return TOL_REL


@pytest.mark.gpu
@pytest.mark.parametrize("n_tokens", [64, 4096], ids=["decode64", "prefill4k"])
def test_triton_grouped_gemm_matches_the_fp32_oracle(n_tokens: int) -> None:
    """The kernel alone, on the rung's own routing — empty groups and all."""
    if not torch.cuda.is_available():
        pytest.skip(_NO_GPU)
    pytest.importorskip("triton", reason=f"Triton absent — {_NO_GPU}")
    from scratch_llm.kernels.moe.grouped_gemm_triton import grouped_gemm_triton

    torch.manual_seed(0)
    n_experts, hidden, inter = SPEC["n_experts"], SPEC["hidden"], SPEC["intermediate"]
    routing = route_topk(torch.randn(n_tokens, n_experts, device="cuda"), SPEC["top_k"])
    p = build_permutation(routing.topk_ids, n_experts)
    a = torch.randn(p.n_rows, hidden, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(n_experts, 2 * inter, hidden, device="cuda") * hidden**-0.5).bfloat16()
    got = grouped_gemm_triton(a, b, p.group_offsets)
    want = reference_grouped_gemm(a, b, p.group_offsets)
    err = (got.float() - want).abs().max().item() / (want.abs().max().item() + 1e-30)
    assert err <= _require_tolerance(), (
        f"max|dC|/max|C| = {err:.4e}. Look for structure before widening this: an error confined "
        f"to the last rows of each group is a tail mask; an error that grows with the expert index "
        f"is an offsets walk; garbage in whole rows is a tile that was never launched."
    )


@pytest.mark.gpu
def test_the_layer_matches_vllm_fused_experts_at_the_identical_routing() -> None:
    """Against the floor itself, fed the same ``topk_ids``/``topk_weights`` — so a difference here
    is arithmetic, never a different choice of experts."""
    if not torch.cuda.is_available():
        pytest.skip(_NO_GPU)
    fused_moe_mod = pytest.importorskip(
        "vllm.model_executor.layers.fused_moe",
        reason="vLLM absent — the floor cannot be the floor. On the box: uv pip install vllm",
    )
    torch.manual_seed(0)
    n_experts, hidden, inter = SPEC["n_experts"], SPEC["hidden"], SPEC["intermediate"]
    x = torch.randn(64, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = (torch.randn(n_experts, 2 * inter, hidden, device="cuda") * hidden**-0.5).bfloat16()
    w2 = (torch.randn(n_experts, hidden, inter, device="cuda") * inter**-0.5).bfloat16()
    routing = route_topk(torch.randn(64, n_experts, device="cuda"), SPEC["top_k"])
    ours = s_r4_fused_moe(x, w1, w2, routing)
    theirs = fused_moe_mod.fused_experts(x, w1, w2, routing.topk_weights, routing.topk_ids)
    err = (ours.float() - theirs.float()).abs().max().item() / (
        theirs.float().abs().max().item() + 1e-30
    )
    assert err <= _require_tolerance(), (
        f"max|dy|/max|y| vs vLLM = {err:.4e}. Part of this is not a bug: vLLM reduces the top_k "
        f"contributions in bf16 (moe_sum, fused_moe.py:1855) and this design reduces in fp32. "
        f"Measure that term on its own before widening the constant."
    )


@pytest.mark.gpu
def test_the_layer_matches_the_fp32_oracle_on_the_box() -> None:
    if not torch.cuda.is_available():
        pytest.skip(_NO_GPU)
    pytest.importorskip("triton", reason=f"Triton absent — {_NO_GPU}")
    torch.manual_seed(0)
    n_experts, hidden, inter = SPEC["n_experts"], SPEC["hidden"], SPEC["intermediate"]
    x = torch.randn(64, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = (torch.randn(n_experts, 2 * inter, hidden, device="cuda") * hidden**-0.5).bfloat16()
    w2 = (torch.randn(n_experts, hidden, inter, device="cuda") * inter**-0.5).bfloat16()
    routing = route_topk(torch.randn(64, n_experts, device="cuda"), SPEC["top_k"])
    assert expert_load(routing.topk_ids, n_experts).n_empty > 0, (
        "the decode point must exercise empty experts"
    )
    ours = s_r4_fused_moe(x, w1, w2, routing)
    want = reference_moe(x, w1, w2, routing)
    err = (ours.float() - want.float()).abs().max().item() / (
        want.float().abs().max().item() + 1e-30
    )
    assert err <= _require_tolerance(), f"max|dy|/max|y| vs fp32 oracle = {err:.4e}"
