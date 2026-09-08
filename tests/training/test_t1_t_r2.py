"""T1/T-R2 — FP8 vs bf16 at fixed tokens. The instrument, not the kernel.

Every assertion here is decidable on a laptop, and every one of them guards a failure that would
otherwise produce a *plausible* number on eight H100s:

  bpb quoted against tokens          uniformly ~4x too large in both arms, cancels in the delta
  bytes of the stream, not the targets   ~0.02% off, forever, in the same direction
  arms drawing different batches     the delta carries a data-order term nobody can subtract
  sigma with ddof=0                  the gate tightens by sqrt(2) at n=2, in the wrong direction
  arms at different token budgets    the longer run wins for a reason that is not its dtype
  fp8 converter matching nothing     a "1.00x speedup, 0.00000 Delta bpb" that reads as success

The gpu tier is the only thing this file cannot settle: whether the fp8 arm's GEMMs are real.

Spec: experiments/T1/T-R2/spec.md   ·   Map: experiments/T1/T-R2/map.md
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from scratch_llm.train import TrainConfig, get_batch, train
from scratch_llm.train import _val_loss as train_val_loss
from scratch_llm.training.bpb import (
    LN2,
    BpbAccounting,
    account,
    bits_per_byte,
    bits_per_byte_from_mean_loss,
    bits_per_token,
    byte_lengths,
)
from scratch_llm.training.fp8_parity import (
    PRESETS,
    ArmConfig,
    RunSpec,
    apply_float8,
    build_model,
    eval_model,
    run_arm,
    to_nn_linear,
)
from scratch_llm.training.run_matrix import (
    ArmResult,
    StepTimer,
    Verdict,
    assert_one_variable,
    batch_schedule,
    batches_from_schedule,
    compare,
    differing_fields,
    digest,
    eval_bpb,
    eval_stream,
    steps_for_tokens,
    verdict,
)
from scratch_llm.training.seed_noise import (
    CHI2_95,
    SeedSigma,
    bpb_gate_tolerance,
    seed_sigma,
    sigma_ci_factors,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOLE_SOURCE = "src/scratch_llm/training/seed_noise.py"


def _hole_is_open():
    """``tests/conftest.py``'s own predicate, loaded BY PATH.

    ``from tests.conftest import hole_is_open`` resolves only under ``python -m pytest``; the bare
    ``pytest`` console script (what CI and the pre-push hook run) leaves rootdir off ``sys.path``.
    A test that passes one way and errors the other is worse than no test, and these two are
    guarding the hole convention itself.
    """
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "_hole_contract", _REPO_ROOT / "tests" / "conftest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hole_is_open


_BOX = (
    "no CUDA — on the box: infra/rent.sh sync-up <user@host> && infra/rent.sh ssh <user@host>, "
    "then: bash ladders/experiments/T1/T-R2/run.sh"
)


def _corpus(n: int = 4096, vocab: int = 64, seed: int = 3) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(seed)).integers(0, vocab, size=n, dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# bpb — the denominator is bytes
# ---------------------------------------------------------------------------------------------


def test_bpb_from_a_known_loss_and_a_known_byte_ratio() -> None:
    """Hand-computed: mean CE = ln 2 nats/token is exactly 1 bit/token. At 4 bytes per token that
    is 1/4 = 0.25 bits per byte. Nothing here is approximate — the arithmetic is exact."""
    acc = BpbAccounting(n_tokens=1000, n_bytes=4000)
    assert acc.bytes_per_token == 4.0
    assert bits_per_byte_from_mean_loss(LN2, acc) == pytest.approx(0.25, abs=1e-12)
    assert bits_per_byte(LN2 * 1000, acc) == pytest.approx(0.25, abs=1e-12)


def test_bpb_is_not_bits_per_token() -> None:
    """The whole trap in one assertion: the same loss gives 1.0 against tokens and 0.25 against
    bytes. A gate at 1e-3 cannot survive a factor of 4, and the factor is invisible in a delta."""
    acc = BpbAccounting(n_tokens=1000, n_bytes=4000)
    assert bits_per_token(LN2) == pytest.approx(1.0)
    assert bits_per_byte_from_mean_loss(LN2, acc) == pytest.approx(0.25)
    assert bits_per_token(LN2) != pytest.approx(bits_per_byte_from_mean_loss(LN2, acc))


def test_bytes_come_from_the_vocab_and_cover_only_the_scored_targets() -> None:
    """``len(vocab[id])`` per token, summed over the TARGETS — not over the whole stream.

    The stream below is 5 tokens (11 bytes); teacher forcing scores 4 transitions whose targets
    are the last 4 tokens (10 bytes). Counting the stream would put 11 in the denominator and make
    bpb 9% too low — the size of the effect this rung is trying to detect, many times over.
    """
    vocab = {0: b"a", 1: b"bb", 2: b"ccc", 3: b"dddd", 4: b"e"}
    lengths = byte_lengths(vocab)
    stream = np.array([0, 1, 2, 3, 4])  # 1+2+3+4+1 = 11 bytes in the stream
    assert int(lengths[stream].sum()) == 11
    acc = account(stream[1:], lengths)  # the four transitions actually scored
    assert (acc.n_tokens, acc.n_bytes) == (4, 10)
    assert acc.n_bytes < int(lengths[stream].sum())


def test_special_tokens_can_be_scored_at_zero_bytes_and_it_moves_bpb() -> None:
    """``<|endoftext|>`` is 13 literal bytes under ``decode`` semantics. Counting them inflates the
    denominator and LOWERS bpb; zeroing them raises it. Both conventions are defensible, the
    difference is not small, and ``compare`` refuses a matrix whose arms disagree."""
    vocab = {0: b"a", 1: b"bb", 2: b"<|endoftext|>"}
    targets = np.array([0, 1, 2, 0, 1, 2])
    counted = account(targets, byte_lengths(vocab))
    zeroed = account(targets, byte_lengths(vocab, zero_byte_ids=[2]), zero_byte_ids=[2])
    assert counted.n_bytes == 2 * (1 + 2 + 13)
    assert zeroed.n_bytes == 2 * (1 + 2)
    assert bits_per_byte(10.0, zeroed) > bits_per_byte(10.0, counted)
    assert zeroed.zero_byte_ids == (2,) and counted.zero_byte_ids == ()


def test_byte_accounting_refuses_a_corpus_the_tokenizer_cannot_explain() -> None:
    lengths = byte_lengths({0: b"a", 1: b"bb"})
    with pytest.raises(ValueError, match="outside the vocab"):
        account(np.array([0, 1, 7]), lengths)


# ---------------------------------------------------------------------------------------------
# data order — one schedule per seed, whatever the arm does with its RNG
# ---------------------------------------------------------------------------------------------


def _record_arm(*, seed: int, corpus: np.ndarray, burn_rng: bool) -> list[torch.Tensor]:
    """Run the REAL training loop for one arm and record every batch it was fed.

    ``burn_rng=True`` stands in for what an fp8 arm does that a bf16 arm does not: consume random
    numbers (torchao draws none today, but a recipe change, a dropout mask, or an init probe would).
    It burns numpy, torch, and stdlib RNG between steps — the three streams
    ``seed_everything`` sets.
    """
    seen: list[torch.Tensor] = []
    spec = RunSpec(tokens=6 * 2 * 16, batch_size=2, context_length=16, preset="debug", device="cpu")
    model, _ = build_model(spec, vocab_size=64, seed=seed)
    schedule = batch_schedule(
        seed,
        steps=spec.steps,
        batch_size=spec.batch_size,
        context_length=spec.context_length,
        corpus_len=len(corpus),
    )
    inner = batches_from_schedule(corpus, schedule, spec.context_length)

    def batch_fn(step: int) -> tuple[torch.Tensor, torch.Tensor]:
        if burn_rng:
            np.random.rand(5)
            torch.randn(5)
            random.random()
        inputs, targets = inner(step)
        seen.append(inputs.clone())
        return inputs, targets

    train(
        TrainConfig(
            max_steps=spec.steps,
            batch_size=spec.batch_size,
            context_length=spec.context_length,
            seed=seed,
            log_every=0,
        ),
        corpus,
        model,
        batch_fn=batch_fn,
    )
    return seen


def test_both_arms_see_the_same_batches_in_the_same_order_at_one_seed() -> None:
    """The paired design's whole premise, checked through the real loop rather than asserted.

    Arm B burns numpy/torch/stdlib randomness at every step. Under ``get_batch`` that alone would
    re-draw its windows (the control below proves it does); under the schedule the two arms are
    element-for-element identical.
    """
    corpus = _corpus()
    a = _record_arm(seed=11, corpus=corpus, burn_rng=False)
    b = _record_arm(seed=11, corpus=corpus, burn_rng=True)
    assert len(a) == len(b) == 6
    for step, (x, y) in enumerate(zip(a, b, strict=True)):
        assert torch.equal(x, y), f"step {step}: the arms diverged"


def test_the_control_the_previous_test_needs_get_batch_really_does_diverge() -> None:
    """Without this, the test above could be passing because nothing in it can ever differ.

    ``train.get_batch`` draws from the GLOBAL numpy RNG at the point of use, so the same
    perturbation that the schedule shrugs off moves its windows. This is the failure mode; the
    schedule is the fix; both are demonstrated.
    """
    corpus = _corpus()
    np.random.seed(11)
    clean, _ = get_batch(corpus, 2, 16)
    np.random.seed(11)
    np.random.rand(5)  # the same burn arm B does
    perturbed, _ = get_batch(corpus, 2, 16)
    assert not torch.equal(clean, perturbed)


def test_different_seeds_give_different_data_orders() -> None:
    kw = dict(steps=8, batch_size=4, context_length=16, corpus_len=4096)
    assert digest(batch_schedule(0, **kw)) != digest(batch_schedule(1, **kw))
    assert digest(batch_schedule(0, **kw)) == digest(batch_schedule(0, **kw))


def test_the_rank_slices_partition_the_global_stream() -> None:
    """world_size re-shards one stream instead of resampling it, so a 1-rank debug run and an
    8-rank run at the same seed consume the same windows in the same step."""
    kw = dict(steps=4, batch_size=2, context_length=16, corpus_len=4096)
    single = batch_schedule(0, world_size=1, **{**kw, "batch_size": 4})
    shards = [batch_schedule(0, world_size=2, rank=r, **kw) for r in range(2)]
    assert np.array_equal(single, np.concatenate(shards, axis=1))


def test_digest_separates_shape_from_content() -> None:
    a = np.arange(12, dtype=np.int64)
    assert digest(a) != digest(a.reshape(3, 4))
    assert digest(a) != digest(a.astype(np.int32))


# ---------------------------------------------------------------------------------------------
# sigma — two seeds, one degree of freedom, and no hiding it
# ---------------------------------------------------------------------------------------------


def test_sigma_is_the_sample_standard_deviation_with_n_minus_one() -> None:
    """Hand-computed, not delegated to numpy: for two values sigma_hat = |x1-x2|/sqrt(2)."""
    s = seed_sigma([0.700, 0.724])
    assert s.n == 2 and s.dof == 1
    assert s.mean == pytest.approx(0.712)
    assert s.sigma == pytest.approx(abs(0.724 - 0.700) / math.sqrt(2))
    assert s.sigma == pytest.approx(float(np.std([0.700, 0.724], ddof=1)))
    assert s.sigma > float(np.std([0.700, 0.724]))  # ddof=0 would tighten the gate by sqrt(2)


def test_two_seeds_is_a_brutal_estimate_and_the_instrument_says_so() -> None:
    """One degree of freedom: the 95% interval for the TRUE sigma runs from 0.45x the estimate to
    32x it. Any gate written as ``k * sigma_hat`` inherits that width, and the argument for k has
    to survive it. The number is asserted here so it appears in the test output, not just prose."""
    lo, hi = sigma_ci_factors(1)
    assert lo == pytest.approx(0.4461, abs=1e-3)
    assert hi > 30.0
    s = seed_sigma([0.700, 0.724])
    ci_lo, ci_hi = s.ci()
    assert ci_lo == pytest.approx(s.sigma * lo) and ci_hi == pytest.approx(s.sigma * hi)
    assert sigma_ci_factors(9)[1] < hi  # more seeds is the only thing that narrows it


def test_chi2_table_matches_scipy_when_scipy_is_installed() -> None:
    """The table is hardcoded so the module imports on a bare box; this is the check that keeps
    the hardcoding honest."""
    chi2 = pytest.importorskip("scipy.stats").chi2
    for dof, (lo, hi) in CHI2_95.items():
        assert lo == pytest.approx(float(chi2.ppf(0.025, dof)), rel=1e-12)
        assert hi == pytest.approx(float(chi2.ppf(0.975, dof)), rel=1e-12)


def test_one_seed_is_not_an_estimate() -> None:
    with pytest.raises(ValueError, match="at least 2 seeds"):
        seed_sigma([0.7])
    with pytest.raises(ValueError, match="non-finite"):
        seed_sigma([0.7, float("nan")])


# ---------------------------------------------------------------------------------------------
# the matrix — one budget, one eval set, one variable
# ---------------------------------------------------------------------------------------------


def _result(arm: str, seed: int, *, tokens: int = 1 << 20, bpb: float = 0.7, **kw) -> ArmResult:
    base = dict(
        steps=16,
        mean_ce=bpb * LN2 * 4,
        tok_s=1000.0,
        tok_s_p25=990.0,
        tok_s_p75=1010.0,
        data_digest=f"data{seed}",
        eval_digest="eval",
        accounting=BpbAccounting(n_tokens=100, n_bytes=400),
        converted_modules=0,
    )
    base.update(kw)
    return ArmResult(arm=arm, seed=seed, tokens=tokens, bpb=bpb, **base)  # type: ignore[arg-type]


def test_the_matrix_refuses_arms_trained_on_different_token_counts() -> None:
    """Fixed tokens is the control. One arm run 1% longer is a better model for a reason that has
    nothing to do with its dtype, and no amount of sigma makes that comparison mean anything."""
    results = [
        _result("bf16", 0, tokens=1 << 20),
        _result("bf16", 1, tokens=1 << 20),
        _result("fp8", 0, tokens=1 << 20),
        _result("fp8", 1, tokens=(1 << 20) + 4096),
    ]
    with pytest.raises(ValueError, match="different token counts"):
        compare(results, baseline="bf16", treatment="fp8")


def test_the_matrix_refuses_a_broken_pairing_a_second_eval_set_and_a_second_convention() -> None:
    ok = [_result(a, s) for a in ("bf16", "fp8") for s in (0, 1)]
    compare(ok, baseline="bf16", treatment="fp8")  # the control: this one is a comparison

    bad_pair = [
        r if r.arm == "bf16" or r.seed else _result("fp8", 0, data_digest="other") for r in ok
    ]
    with pytest.raises(ValueError, match="different data orders"):
        compare(bad_pair, baseline="bf16", treatment="fp8")

    bad_eval = [r if r.arm == "bf16" else _result(r.arm, r.seed, eval_digest="other") for r in ok]
    with pytest.raises(ValueError, match="different eval streams"):
        compare(bad_eval, baseline="bf16", treatment="fp8")

    other_acc = BpbAccounting(n_tokens=100, n_bytes=300, zero_byte_ids=(2,))
    bad_acc = [r if r.arm == "bf16" else _result(r.arm, r.seed, accounting=other_acc) for r in ok]
    with pytest.raises(ValueError, match="different byte accounting"):
        compare(bad_acc, baseline="bf16", treatment="fp8")

    with pytest.raises(ValueError, match="incomplete matrix"):
        compare(ok[:3], baseline="bf16", treatment="fp8")


def test_the_comparison_reports_paired_and_unpaired_deltas() -> None:
    """The arms share a data order per seed, so the per-seed deltas are paired and the pairing
    removes the data-order term. Both numbers are exposed; which one the gate uses is Huy's."""
    results = [
        _result("bf16", 0, bpb=0.700),
        _result("bf16", 1, bpb=0.724),
        _result("fp8", 0, bpb=0.706),
        _result("fp8", 1, bpb=0.728),
    ]
    cmp_ = compare(results, baseline="bf16", treatment="fp8")
    assert cmp_.paired_deltas == pytest.approx((0.006, 0.004))
    assert cmp_.delta_bpb_paired == pytest.approx(0.005)
    assert cmp_.delta_bpb_means == pytest.approx(0.005)
    assert cmp_.baseline_sigma.dof == 1
    assert cmp_.speedup == pytest.approx(1.0)


def test_token_budget_must_be_a_whole_number_of_steps() -> None:
    assert steps_for_tokens(1024, batch_size=2, context_length=16, world_size=2) == 16
    with pytest.raises(ValueError, match="not a whole number of steps"):
        steps_for_tokens(1025, batch_size=2, context_length=16, world_size=2)


def test_the_arms_differ_in_exactly_one_field() -> None:
    bf16, fp8 = ArmConfig(precision="bf16"), ArmConfig(precision="fp8")
    assert differing_fields(bf16, fp8) == ["precision"]
    assert_one_variable(bf16, fp8, "precision")
    with pytest.raises(ValueError, match="exactly one field"):
        assert_one_variable(bf16, ArmConfig(precision="fp8", max_lr=1e-3), "precision")


def test_the_step_timer_drops_the_warmup_and_demands_fifty_timed_steps() -> None:
    timer = StepTimer(lambda step: (torch.zeros(1), torch.zeros(1)))
    for step in range(30):
        timer(step)
    assert len(timer.intervals_s) == 29  # the first call starts the clock and times nothing
    with pytest.raises(ValueError, match="need 50"):
        timer.summary(tokens_per_step=1024, warmup=20, min_timed=50)
    s = timer.summary(tokens_per_step=1024, warmup=2, min_timed=4)
    assert s["tok_s_p25"] <= s["tok_s"] <= s["tok_s_p75"]


# ---------------------------------------------------------------------------------------------
# the arms are the same experiment up to one cast
# ---------------------------------------------------------------------------------------------


def test_both_arms_start_from_bitwise_identical_weights_at_one_seed() -> None:
    """'One variable' includes the starting point. Same seed, same tensors; different seed, not."""
    spec = RunSpec(tokens=64, batch_size=2, context_length=16, preset="debug")
    a, _ = build_model(spec, vocab_size=64, seed=5)
    b, _ = build_model(spec, vocab_size=64, seed=5)
    c, _ = build_model(spec, vocab_size=64, seed=6)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    assert not all(torch.equal(v, c.state_dict()[k]) for k, v in a.state_dict().items())


def test_the_nn_linear_shim_shares_parameters_and_keeps_the_forward() -> None:
    """torchao matches on the class, and ``scratch_llm.model.Linear`` is not ``nn.Linear`` — so
    both arms get this surgery, and the shim's own dispatch difference cancels in the delta
    instead of being attributed to fp8."""
    from scratch_llm.model import TransformerLM

    cfg = PRESETS["debug"].model_config(vocab_size=64, context_length=16)
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    ids = torch.randint(0, 64, (2, 16))
    before = model(ids)
    weights = {id(p) for p in model.parameters()}
    n = to_nn_linear(model)
    per_block = PRESETS["debug"].linears_per_block
    assert n == cfg.n_layers * per_block + 1  # 4 attention + 3 SwiGLU per block, plus lm_head
    assert {id(p) for p in model.parameters()} == weights  # same Parameter objects, no copies
    assert all(isinstance(m, torch.nn.Linear) for m in model.modules() if hasattr(m, "in_features"))
    torch.testing.assert_close(model(ids), before, rtol=1e-5, atol=1e-5)


def test_both_arms_are_scored_through_the_same_unquantized_module() -> None:
    """The fp8 arm's own forward quantizes; scoring it in place would fold an fp8 INFERENCE effect
    into a number the rung reports as an fp8 TRAINING effect."""
    spec = RunSpec(tokens=64, batch_size=2, context_length=16, preset="debug")
    trained, cfg = build_model(spec, vocab_size=64, seed=1)
    scorer = eval_model(trained, cfg, spec)
    assert type(scorer) is type(trained)
    for k, v in trained.state_dict().items():
        assert torch.equal(v, scorer.state_dict()[k]), k
    assert all(p.dtype == torch.float32 for p in scorer.parameters())


def test_eval_bpb_scores_the_same_windows_as_the_training_loops_val_hook() -> None:
    """The eval stream is the loop's own val rule, so an arm's reported bpb and the curve it was
    watched on are the same measurement. Divergence here would be silent."""
    spec = RunSpec(tokens=64, batch_size=2, context_length=16, preset="debug")
    model, _ = build_model(spec, vocab_size=64, seed=2)
    val = _corpus(n=512, vocab=64, seed=9)
    stream = eval_stream(val, 16, 4)
    lengths = np.full(64, 3, dtype=np.int64)
    score, mean_ce, acc = eval_bpb(model, stream, lengths, windows_per_forward=4)
    assert mean_ce == pytest.approx(train_val_loss(model, val, 16, 4, "cpu"), rel=1e-5)
    assert acc.n_tokens == 4 * 16 and acc.n_bytes == 4 * 16 * 3
    assert score == pytest.approx(mean_ce / LN2 / 3.0, rel=1e-6)


def test_eval_stream_is_deterministic_and_consumes_no_random_state() -> None:
    val = _corpus(n=512, vocab=64, seed=9)
    np.random.seed(4)
    before = np.random.get_state()[2]  # type: ignore[index]
    a, b = eval_stream(val, 16, 4), eval_stream(val, 16, 4)
    assert a.digest == b.digest
    assert np.random.get_state()[2] == before  # type: ignore[index]


# ---------------------------------------------------------------------------------------------
# the hole
# ---------------------------------------------------------------------------------------------


@pytest.mark.hole("T1/T-R2", _HOLE_SOURCE)
def test_the_bpb_tolerance_is_a_function_of_the_measured_seed_noise() -> None:
    """Fails while ``bpb_gate_tolerance`` raises; passes once Huy decides it.

    The assertions are walls, not the decision. They do not say what k is, whether the gate is on
    the paired or the unpaired delta, or whether sigma_hat or a CI bound multiplies it — every one
    of those is the rung's argument. They say only:

    * a tolerance is a finite positive width;
    * a noisier baseline cannot produce a STRICTER gate (monotone in sigma_hat);
    * the answer must MOVE when sigma_hat moves. This is the one that matters: it rejects a
      prior recalled from plan section 05 — measured on another card, another model, another
      corpus — dressed up as a measurement;
    * zero degrees of freedom is not an estimate, so it is not a tolerance either.
    """
    small = seed_sigma([0.700, 0.706])
    large = seed_sigma([0.700, 0.712])  # exactly 2x the sigma, same mean
    assert large.sigma == pytest.approx(2 * small.sigma)

    tol_small = bpb_gate_tolerance(small)
    tol_large = bpb_gate_tolerance(large)
    assert math.isfinite(tol_small) and tol_small > 0.0
    assert tol_large >= tol_small
    assert tol_large != pytest.approx(tol_small), (
        "the tolerance did not respond to the measured sigma — a constant is not an estimate"
    )
    with pytest.raises((ValueError, ZeroDivisionError, KeyError)):
        bpb_gate_tolerance(SeedSigma(values=(0.7,), mean=0.7, sigma=0.0))


def test_the_statistics_stay_computable_while_the_hole_is_open() -> None:
    """``compare`` must never touch the tolerance: the CPU self-test, the floor run, and the
    printed sigma all have to work on the day before Huy fills it — and on the day after.

    Two-state, like every hole-adjacent assertion here (``tests/conftest.py``): while the hole is
    open ``verdict`` raises and nothing above it does; once filled, ``verdict`` returns and the
    statistics are still computed without it.
    """
    results = [_result(a, s, bpb=0.70 + 0.01 * s) for a in ("bf16", "fp8") for s in (0, 1)]
    cmp_ = compare(results, baseline="bf16", treatment="fp8")
    assert cmp_.baseline_sigma.sigma > 0
    assert cmp_.delta_bpb_paired == pytest.approx(0.0)
    if _hole_is_open()(_HOLE_SOURCE):
        with pytest.raises(NotImplementedError, match="HUY"):
            verdict(cmp_)
    else:
        assert isinstance(verdict(cmp_), Verdict)


def test_the_hole_sentinel_survives_the_formatter() -> None:
    """``tests/conftest.py`` decides a hole is open by matching ``NotImplementedError("HUY:`` as a
    regex across whitespace. ``ruff format`` wraps that call onto two lines, which deletes the
    literal substring — a substring check would then report the hole CLOSED, the strict xfail would
    never be applied, and the guard above would go red while the tolerance is still unwritten.

    Two-state: a source that no longer raises has no sentinel to lose, so the second disjunct is
    what holds after Huy fills it. (Same shape as ``tests/kernels/attention/test_k2_a_r3.py``.)
    """
    source = (_REPO_ROOT / _HOLE_SOURCE).read_text(encoding="utf-8")
    assert _hole_is_open()(_HOLE_SOURCE) or "raise NotImplementedError" not in source, (
        "the source still raises NotImplementedError but conftest.hole_is_open() says the hole is "
        'closed — the `NotImplementedError("HUY: ...")` sentinel has been split across lines'
    )


def test_the_plan_prior_is_not_a_constant_anywhere_in_the_instrument() -> None:
    """Plan section 05 quotes a prior for this sigma. Turning that expectation into a constant is
    exactly the move the seal exists to prevent: it was measured on a different card, model, and
    corpus, and here it would silently become the gate. The literal is spelled out of two halves so
    the check cannot pass by accident on a file that contains it."""
    root = _REPO_ROOT / "src" / "scratch_llm" / "training"
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "0." + "012" not in text, f"{path.name} hardcodes the plan's prior sigma"


# ---------------------------------------------------------------------------------------------
# gpu — the only tier that can tell an fp8 arm from a bf16 one
# ---------------------------------------------------------------------------------------------


@pytest.mark.gpu
def test_float8_conversion_swaps_a_known_number_of_linears() -> None:
    if not torch.cuda.is_available():
        pytest.skip(_BOX)
    pytest.importorskip(
        "torchao",
        reason="fp8 arm needs torchao: USE_CPP=0 pip install git+https://github.com/pytorch/ao.git",
    )
    spec = RunSpec(tokens=1 << 14, batch_size=2, context_length=128, preset="llama3_1b")
    model, cfg = build_model(spec, vocab_size=1024, seed=0)
    swapped = apply_float8(model, recipe="rowwise", filter_fqns=("lm_head",))
    assert swapped == cfg.n_layers * PRESETS["llama3_1b"].linears_per_block, (
        "every block linear but lm_head should be fp8"
    )
    assert not any(
        type(m).__name__ == "Float8Linear" for n, m in model.named_modules() if "lm_head" in n
    )


@pytest.mark.gpu
def test_a_converter_that_matches_nothing_is_refused() -> None:
    if not torch.cuda.is_available():
        pytest.skip(_BOX)
    pytest.importorskip("torchao", reason="fp8 arm needs torchao")
    spec = RunSpec(tokens=1 << 14, batch_size=2, context_length=128, preset="debug")
    model, _ = build_model(spec, vocab_size=1024, seed=0)
    with pytest.raises(RuntimeError, match="swapped 0 modules"):
        apply_float8(
            model,
            recipe="rowwise",
            filter_fqns=("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3", "lm_head"),
        )


@pytest.mark.gpu
def test_one_cell_of_the_matrix_runs_end_to_end_on_the_box() -> None:
    if not torch.cuda.is_available():
        pytest.skip(_BOX)
    spec = RunSpec(
        tokens=80 * 2 * 128,
        batch_size=2,
        context_length=128,
        preset="debug",
        device="cuda",
        eval_windows=2,
        timing_warmup=20,
        timing_min_steps=50,
    )
    corpus = _corpus(n=1 << 15, vocab=256, seed=1)
    lengths = np.full(256, 2, dtype=np.int64)
    r = run_arm(
        spec,
        ArmConfig(precision="bf16"),
        seed=0,
        train_data=corpus[:-4096],
        val_data=corpus[-4096:],
        lengths=lengths,
    )
    assert r.tok_s > 0 and math.isfinite(r.bpb)
    assert r.accounting.bytes_per_token == pytest.approx(2.0)
