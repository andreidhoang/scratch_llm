"""S1/S-R3 — bucketed CUDA-graph dispatch, per-position acceptance, and the model's guard.

Four tiers, and only the first three run today:

  CPU/dispatch   bucket selection and the decoder's dispatch path, driven with FAKE runners. This
                 is where the cross-request bug lives, and faking the runner is the only way to
                 catch it on a machine with no GPU — the real failure is silent on a GPU too.
  CPU/acceptance the per-position bookkeeping: what "rejected" means for a position that was never
                 compared, and the demonstration that the flat average cannot tell two runs apart
                 that the profile separates.
  CPU/simulator  the brute-force round simulator itself. It is the oracle for the hole, so it must
                 be checked by something that is not the hole — otherwise a bug in the simulator
                 becomes Huy's problem on the morning he fills the model.
  hole           the model against the simulator. Strict-xfail while `spec_round_model` raises
                 (tests/conftest.py); an ordinary test the moment it is filled.
  gpu            the graph arm is a wall-clock claim and cannot be made on this box; the marked
                 test skips with the command that makes it on the rented one.

Spec: experiments/S1/S-R3/spec.md   ·   Map: experiments/S1/S-R3/map.md
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence

import pytest
import torch

from scratch_llm.serving.acceptance import (
    PositionAcceptance,
    break_even_draft_length,
    spec_round_model,
    speedup_curve,
)
from scratch_llm.serving.graph_buckets import (
    BucketedGraphDecoder,
    default_decode_buckets,
    select_bucket,
    validate_buckets,
)

#: The command that produces the graph number. Every GPU skip in this file quotes it verbatim, so
#: a skipped test is a work order and not a shrug.
BOX_CMD = (
    "infra/rent.sh sync-up <user@host> && ssh in && cd ladders && \\\n"
    "        bash experiments/S1/S-R3/run.sh                                    # graphs\n"
    "        S_R3_MODE=spec S_R3_CKPT=runs/f1/model_final.pt \\\n"
    "            bash experiments/S1/S-R3/run.sh                                # speculation\n"
    "        .venv/bin/python -m pytest tests/serving/test_s1_s_r3.py -m gpu    # this test"
)


# =================================================================================================
# The brute-force round simulator — the hole's oracle, independent of any closed form
# =================================================================================================


def simulate_expected_accepted(alpha: Sequence[float]) -> float:
    """E[# accepted draft tokens in one round], by exhaustive enumeration. No formula.

    Enumerates every accept/reject pattern over ``len(alpha)`` positions, weights it by the product
    of its per-position probabilities, and counts the accepted prefix PROCEDURALLY — by walking the
    pattern and stopping at the first reject, exactly as ``speculative_generate``'s accept loop
    does. Nothing here knows what the answer looks like in closed form, which is the point: it will
    catch a model that is right on a flat profile and wrong on a shaped one.

    Patterns whose later bits are unreachable (anything after the first reject) are still summed
    over both branches. That is not a bug — marginalising a position the round never tested
    contributes a factor of ``alpha_i + (1 - alpha_i) = 1``, so the enumeration stays a proper
    expectation while never having to special-case truncation.

    Exponential in ``len(alpha)`` on purpose. A Monte-Carlo estimate would need a sampling
    tolerance, and choosing tolerances is not the agent's to do on this rung; enumeration is exact,
    deterministic, and fast enough for the draft lengths speculation is actually run at.
    """
    if not alpha:
        raise ValueError("alpha must be non-empty")
    if any(not 0.0 <= a <= 1.0 for a in alpha):
        raise ValueError(f"alpha entries must be probabilities, got {list(alpha)}")
    total = 0.0
    for pattern in itertools.product((0, 1), repeat=len(alpha)):
        prob = 1.0
        for a, bit in zip(alpha, pattern, strict=True):
            prob *= a if bit else (1.0 - a)
        accepted = 0
        for bit in pattern:  # the prefix rule, walked rather than solved
            if bit == 0:
                break
            accepted += 1
        total += prob * accepted
    return total


def _random_profile(rng: random.Random, k: int) -> list[float]:
    return [rng.random() for _ in range(k)]


# =================================================================================================
# CPU/simulator — the oracle checked against things that are true by inspection
# =================================================================================================


def test_simulator_accepts_everything_when_every_position_always_lands() -> None:
    """alpha = all ones: no position ever rejects, so every round accepts the whole draft."""
    for k in range(1, 7):
        assert simulate_expected_accepted([1.0] * k) == pytest.approx(float(k))


def test_simulator_accepts_nothing_when_the_first_position_always_fails() -> None:
    """alpha[0] = 0 truncates the round before position 1, whatever the rest of the profile says.

    This is the assertion that separates a prefix-accept model from a per-position one: the tail is
    perfect and contributes exactly zero.
    """
    assert simulate_expected_accepted([0.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_simulator_is_monotone_in_every_position() -> None:
    """Raising any single alpha_i can only raise E[accepted] — it adds probability to longer runs."""
    rng = random.Random(7)
    base = _random_profile(rng, 5)
    base_val = simulate_expected_accepted(base)
    for i in range(len(base)):
        raised = list(base)
        raised[i] = min(1.0, raised[i] + 0.25)
        assert simulate_expected_accepted(raised) >= base_val - 1e-12


def test_the_flat_average_of_a_profile_does_not_determine_the_round() -> None:
    """Two profiles, same mean alpha, different E[accepted]. The rung's thesis, as arithmetic.

    A front-loaded profile keeps the round alive long enough to reach its later positions; a
    back-loaded one wastes its good positions behind a bad gate. Any model that consumes only the
    mean must give these two the same answer, and this test says that answer is wrong.
    """
    front = [0.9, 0.9, 0.1, 0.1]
    back = [0.1, 0.1, 0.9, 0.9]
    assert sum(front) == pytest.approx(sum(back))
    assert simulate_expected_accepted(front) > simulate_expected_accepted(back)


# =================================================================================================
# CPU/acceptance — the per-position bookkeeping
# =================================================================================================


def test_a_rejected_position_and_everything_after_it_is_rejected() -> None:
    """One round: 4 drafted, 1 accepted. Position 1 was TESTED and lost; 2 and 3 never ran.

    The distinction is the instrument's whole reason to exist. Position 1 is evidence about the
    drafter; positions 2 and 3 are evidence about position 1. Folding them together is what makes
    an averaged acceptance rate decay with K for reasons that have nothing to do with K.
    """
    acc = PositionAcceptance(draft_len=4)
    acc.observe_round(n_drafted=4, n_accepted=1)

    assert acc.offered == [1, 1, 1, 1], "all four guesses existed"
    assert acc.reached == [1, 1, 0, 0], "position 1 was compared and rejected; 2-3 never were"
    assert acc.accepted == [1, 0, 0, 0]
    assert acc.n_rounds == 1

    cond = acc.conditional()
    assert cond[0] == pytest.approx(1.0)
    assert cond[1] == pytest.approx(0.0), "tested and failed is a real zero"
    assert cond[2] is None and cond[3] is None, "never tested is not a zero — it is no evidence"

    assert acc.cumulative() == [1.0, 0.0, 0.0, 0.0]
    assert acc.mean_accepted_per_round == pytest.approx(1.0)


def test_a_fully_accepted_round_reaches_every_position_and_rejects_none() -> None:
    acc = PositionAcceptance(draft_len=3)
    acc.observe_round(3, 3)
    assert acc.offered == [1, 1, 1]
    assert acc.reached == [1, 1, 1]
    assert acc.accepted == [1, 1, 1]
    assert acc.conditional() == [1.0, 1.0, 1.0]


def test_a_round_rejected_at_position_zero_reaches_only_position_zero() -> None:
    acc = PositionAcceptance(draft_len=4)
    acc.observe_round(4, 0)
    assert acc.reached == [1, 0, 0, 0]
    assert acc.accepted == [0, 0, 0, 0]
    assert acc.conditional() == [0.0, None, None, None]


def test_a_short_draft_offers_fewer_positions_than_the_configured_k() -> None:
    """``NGramDrafter`` returns [] on a lookup miss and a short list near the end of a match."""
    acc = PositionAcceptance(draft_len=4)
    acc.observe_round(2, 2)
    assert acc.offered == [1, 1, 0, 0], "positions 2-3 were never proposed at all"
    assert acc.reached == [1, 1, 0, 0], "a fully accepted short draft rejects nothing"
    assert acc.flat_acceptance_rate == pytest.approx(1.0)


def test_the_flat_rate_cannot_distinguish_two_runs_the_profile_separates() -> None:
    """Same ``flat_acceptance_rate``, different conditional profiles — the measured version of the
    simulator test above, and the reason this rung reports per position."""
    a = PositionAcceptance(draft_len=4)
    a.observe_round(4, 4)
    a.observe_round(4, 0)

    b = PositionAcceptance(draft_len=4)
    b.observe_round(4, 2)
    b.observe_round(4, 2)

    assert a.flat_acceptance_rate == pytest.approx(b.flat_acceptance_rate) == pytest.approx(0.5)
    assert a.conditional() != b.conditional()


def test_the_bookkeeping_refuses_an_impossible_round() -> None:
    acc = PositionAcceptance(draft_len=4)
    with pytest.raises(ValueError, match="n_accepted"):
        acc.observe_round(n_drafted=2, n_accepted=3)
    with pytest.raises(ValueError, match="draft_len"):
        acc.observe_round(n_drafted=5, n_accepted=0)
    assert acc.n_rounds == 0, "a refused round must not be counted"


def test_an_empty_run_reports_no_evidence_rather_than_zeros() -> None:
    acc = PositionAcceptance(draft_len=3)
    assert acc.conditional() == [None, None, None]
    assert acc.cumulative() == [0.0, 0.0, 0.0]
    assert acc.flat_acceptance_rate == 0.0


# =================================================================================================
# CPU/dispatch — bucket selection and the wrong-shape replay
# =================================================================================================

BUCKETS = (8, 16, 32)


class _FakeGraphRunner:
    """A captured graph, faked: fixed width, deterministic output, counts its replays."""

    def __init__(self, bucket: int, *, returns: int | None = None) -> None:
        self.bucket = bucket
        self.width = bucket if returns is None else returns
        self.calls = 0
        self.last_input: torch.Tensor | None = None

    def __call__(self, staging: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_input = staging.clone()
        assert staging.shape[0] == self.bucket, "the decoder handed a capture the wrong width"
        return (staging + 1000)[: self.width]


class _FakeEager:
    """The eager fallback: any width, and a different output signature so the arms are separable."""

    def __init__(self) -> None:
        self.calls = 0
        self.widths: list[int] = []

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.widths.append(int(tokens.shape[0]))
        return tokens + 7


def test_select_bucket_takes_the_smallest_bucket_at_or_above_the_batch() -> None:
    """Exactly at a bucket boundary the batch takes that bucket, not the next one up: an off-by-one
    the other way pads every full batch and pays for a row it does not have."""
    assert select_bucket(1, BUCKETS) == 8
    assert select_bucket(7, BUCKETS) == 8
    assert select_bucket(8, BUCKETS) == 8
    assert select_bucket(9, BUCKETS) == 16
    assert select_bucket(16, BUCKETS) == 16
    assert select_bucket(17, BUCKETS) == 32
    assert select_bucket(32, BUCKETS) == 32


def test_select_bucket_never_returns_a_bucket_smaller_than_the_batch() -> None:
    """The invariant, over every batch in range, rather than over the handful of cases above."""
    for batch in range(1, max(BUCKETS) + 1):
        chosen = select_bucket(batch, BUCKETS)
        assert chosen is not None
        assert chosen >= batch, f"batch {batch} was given bucket {chosen} — rows would be dropped"


def test_a_batch_above_the_largest_bucket_is_not_a_bucket() -> None:
    """``None`` means eager. Clamping to the largest bucket here is the whole bug."""
    for batch in (33, 40, 64, 1000):
        assert select_bucket(batch, BUCKETS) is None


def test_select_bucket_refuses_an_empty_batch_and_a_malformed_ladder() -> None:
    with pytest.raises(ValueError, match="batch"):
        select_bucket(0, BUCKETS)
    with pytest.raises(ValueError, match="sorted"):
        select_bucket(4, (32, 8, 16))
    with pytest.raises(ValueError, match="sorted"):
        select_bucket(4, (8, 8, 16))
    with pytest.raises(ValueError, match=">= 1"):
        select_bucket(4, (0, 8))
    assert validate_buckets([1, 2, 4]) == (1, 2, 4)


def test_a_batch_above_the_largest_bucket_falls_back_to_eager_and_replays_no_graph() -> None:
    """THE BUG: a 40-row batch must not replay the 32-row graph.

    If it did, torch would not complain. The staging vector is a real ``(32,)`` tensor, the replay
    returns ``(32,)``, and ``out[:40]`` on it silently yields 32 rows — so the caller zips 32 tokens
    against 40 request ids and request 32 onward receive tokens computed for someone else's KV
    slot on the previous step. No exception, no NaN, no wrong-looking logits: just another
    request's text. The only safe answer above the ladder is eager, and this test pins all three
    consequences — eager ran, the graph did not, and the result is 40 rows wide.
    """
    runners = {b: _FakeGraphRunner(b) for b in BUCKETS}
    eager = _FakeEager()
    dec = BucketedGraphDecoder(BUCKETS, lambda b: runners[b], eager)
    dec.capture_all()

    tokens = torch.arange(40, dtype=torch.long)
    out = dec.step(tokens)

    assert eager.calls == 1 and eager.widths == [40]
    assert all(r.calls == 0 for r in runners.values()), "no graph may be replayed above the ladder"
    assert out.shape[0] == 40, "every row of the batch must come back"
    assert torch.equal(out, tokens + 7), "the tokens are eager's, not a graph's"
    assert dec.stats() == {
        "graph_steps": 0,
        "eager_steps": 1,
        "padded_rows": 0,
        "captured_buckets": 3,
    }


def test_a_batch_inside_the_ladder_replays_its_bucket_padded_and_slices_back() -> None:
    """Batch 5 -> bucket 8: the graph sees 8 rows, the caller gets its 5 back, unmixed."""
    runners = {b: _FakeGraphRunner(b) for b in BUCKETS}
    eager = _FakeEager()
    dec = BucketedGraphDecoder(BUCKETS, lambda b: runners[b], eager, pad_token=0)

    tokens = torch.tensor([11, 12, 13, 14, 15], dtype=torch.long)
    out = dec.step(tokens)

    assert runners[8].calls == 1 and runners[16].calls == 0 and runners[32].calls == 0
    assert eager.calls == 0
    staged = runners[8].last_input
    assert staged is not None and staged.shape[0] == 8
    assert torch.equal(staged[:5], tokens)
    assert torch.equal(staged[5:], torch.zeros(3, dtype=torch.long)), "the tail is pad, not stale"
    assert torch.equal(out, tokens + 1000)
    assert dec.stats()["padded_rows"] == 3, "3 rows of model work were done and thrown away"


def test_the_staging_buffer_does_not_leak_the_previous_step_into_the_pad_rows() -> None:
    """A shrinking batch must not leave the previous step's tokens in the padded tail.

    Those rows still run the model. Leaving a real token there computes attention over another
    request's context, which is how a padded slot turns into a source of cross-request state even
    when the output slice is correct.
    """
    runners = {b: _FakeGraphRunner(b) for b in BUCKETS}
    dec = BucketedGraphDecoder(BUCKETS, lambda b: runners[b], _FakeEager(), pad_token=0)
    dec.step(torch.arange(1, 9, dtype=torch.long))  # a full bucket-8 step
    dec.step(torch.tensor([99, 98], dtype=torch.long))  # then a 2-row step in the same bucket
    staged = runners[8].last_input
    assert staged is not None
    assert torch.equal(staged, torch.tensor([99, 98, 0, 0, 0, 0, 0, 0], dtype=torch.long))


def test_a_runner_that_returns_the_wrong_width_raises_instead_of_being_sliced() -> None:
    """The last line of defence: if a capture and its replay ever disagree, fail loudly."""
    bad = _FakeGraphRunner(8, returns=6)
    dec = BucketedGraphDecoder((8,), lambda _b: bad, _FakeEager())
    with pytest.raises(RuntimeError, match="another request's tokens"):
        dec.step(torch.arange(5, dtype=torch.long))


def test_capture_runs_largest_bucket_first() -> None:
    """The caching allocator serves the small captures out of the large one's pool; the order is a
    memory decision, not a stylistic one (vLLM does the same, gpu_model_runner.py:6987-6989)."""
    order: list[int] = []

    def capture(bucket: int) -> _FakeGraphRunner:
        order.append(bucket)
        return _FakeGraphRunner(bucket)

    BucketedGraphDecoder(BUCKETS, capture, _FakeEager()).capture_all()
    assert order == [32, 16, 8]


def test_the_default_ladder_covers_the_max_batch_and_stays_ascending() -> None:
    """The largest batch the scheduler admits must be a bucket, on-stride or not — it is exactly
    the batch that must not fall to eager."""
    for max_batch in (1, 5, 8, 32, 33, 64):
        ladder = default_decode_buckets(max_batch)
        assert ladder == tuple(sorted(set(ladder)))
        assert ladder[-1] == max_batch
        assert select_bucket(max_batch, ladder) == max_batch


# =================================================================================================
# The hole guard — exactly one (tests/conftest.py turns it into a strict xfail while open)
# =================================================================================================


@pytest.mark.hole("S1/S-R3", "src/scratch_llm/serving/acceptance.py")
def test_the_speedup_model_agrees_with_the_brute_force_round_simulation() -> None:
    """Fails while ``spec_round_model`` raises; passes when it reproduces the enumeration.

    ``expected_accepted`` is checked against a VALUE, because the simulator computes that value
    exactly and independently: enumerate every accept/reject pattern, weight it by the profile,
    walk the prefix rule. There is no tolerance to design here — both sides are finite sums over
    the same probabilities, so the only disagreements possible are floating-point last bits (which
    ``math.isclose``'s language default absorbs) or a different model, which is the thing being
    tested.

    ``speedup`` is checked only by DIRECTION. Its value depends on a cost decision the hole's
    comment explicitly leaves open — whether the verify forward over 1 + K positions costs the same
    as a one-token decode forward — so pinning a number here would be the agent choosing the model.
    What every correct model must satisfy is asserted instead: more expensive drafts are never
    better, and free drafts that always land are never worse than not speculating.
    """
    rng = random.Random(20260907)
    profiles = [
        [1.0],
        [0.5, 0.5],
        [0.9, 0.9, 0.1, 0.1],  # front-loaded
        [0.1, 0.1, 0.9, 0.9],  # back-loaded, same mean
        [0.0, 1.0, 1.0, 1.0],  # gated shut at position 0
        [1.0, 1.0, 1.0, 1.0],
        *[_random_profile(rng, k) for k in (1, 2, 3, 4, 5, 6, 7, 8)],
    ]

    for alpha in profiles:
        got = spec_round_model(alpha, cost_ratio=0.0)
        assert got.draft_len == len(alpha)
        assert math.isclose(got.expected_accepted, simulate_expected_accepted(alpha)), (
            f"E[accepted] disagrees with the enumeration on {alpha}: model "
            f"{got.expected_accepted!r} vs simulation {simulate_expected_accepted(alpha)!r}"
        )
        assert got.speedup > 0.0 and math.isfinite(got.speedup)

    # Direction 1: drafting costs something, and more of it is never an improvement.
    shaped = [0.8, 0.6, 0.4, 0.2]
    cheap = spec_round_model(shaped, cost_ratio=0.0).speedup
    dear = spec_round_model(shaped, cost_ratio=0.5).speedup
    assert dear <= cheap, "a more expensive drafter cannot make the same profile faster"

    # Direction 2: a strictly better profile at the same cost is never slower.
    worse = spec_round_model([0.4, 0.3, 0.2, 0.1], cost_ratio=0.1).speedup
    better = spec_round_model([0.9, 0.8, 0.7, 0.6], cost_ratio=0.1).speedup
    assert better >= worse

    # Direction 3: free drafts that always land beat plain decode. Any model that fails this has
    # its round accounting inverted somewhere.
    assert spec_round_model([1.0] * 4, cost_ratio=0.0).speedup >= 1.0

    # The plumbing over the model: the scan, and the crossing read off it.
    curve = speedup_curve(shaped, cost_ratio=0.1)
    assert [r.draft_len for r in curve] == [1, 2, 3, 4]
    k_star = break_even_draft_length(shaped, cost_ratio=0.1)
    assert 0 <= k_star <= len(shaped)
    if k_star:
        assert curve[k_star - 1].speedup >= 1.0
        assert all(r.speedup < 1.0 for r in curve[k_star:])


# =================================================================================================
# gpu — the wall-clock half, which this box cannot make
# =================================================================================================


@pytest.mark.gpu
def test_graph_replay_is_token_identical_to_eager_decode() -> None:
    """Capture changes HOW the step is launched, never WHAT it computes.

    This is the correctness gate the graph number sits on: if the bucketed replay's tokens differ
    from the eager path's, the speedup is measuring a different computation and the rung is void.
    It needs the paged Triton kernel and a real capture, so it runs on the box, not here.
    """
    if not torch.cuda.is_available():
        pytest.skip(
            f"no CUDA device — the graph arm is a wall-clock claim. On the box:\n    {BOX_CMD}"
        )
    pytest.importorskip("triton")

    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "bench"))
    from scratch_llm.serving.graph_buckets import BucketedGraphDecoder, paged_capture_factory
    from serving.s1_ladder import GraphShape, _build_model, _eager_step_fn, _prefilled

    shape = GraphShape(batch=5)  # off-bucket on purpose: the padded path must stay token-exact
    model, cfg = _build_model("cuda")

    eager_cache, first = _prefilled(model, cfg, shape.batch, shape.prompt, "cuda")
    eager_step, state = _eager_step_fn(model, eager_cache)
    state["last"] = first
    eager_tokens = torch.stack([eager_step() for _ in range(8)])

    dec = BucketedGraphDecoder(
        (8,),
        paged_capture_factory(model, lambda b: _prefilled(model, cfg, b, shape.prompt, "cuda")),
        eager_step,
        device="cuda",
    )
    dec.capture_all()
    last = first.clone()
    graph_tokens = []
    for _ in range(8):
        last = dec.step(last)
        graph_tokens.append(last)

    assert torch.equal(eager_tokens, torch.stack(graph_tokens))
    assert dec.stats()["graph_steps"] == 8 and dec.stats()["eager_steps"] == 0
