"""T1/T-R4 — the per-checkpoint report card and the vibe-eval harness (CPU only).

Four claims, one test group each, all of them about *comparability* rather than about scores:

1. **Refusal.** Two cards evaluated on different prompt sets, different decoding configs or
   different held-out splits cannot be compared, and the refusal names the field that differs.
2. **Determinism.** The same checkpoint evaluated twice — including after a round-trip through
   ``save_checkpoint``/``build_model_from_checkpoint``, and at temperature > 0 — yields
   *byte-identical token ids*, and the run is invariant to prompt order.
3. **No blank cells.** A metric a card's stage requires and does not have is an error at
   construction; ``card.metric(...)`` raises rather than returning ``None``.
4. **Exactness.** ``d20_gate`` passes at equality and fails one ulp beyond, clause by clause.

Every threshold appearing below is a **synthetic boundary fixture** (0.5, 1.0, 10) chosen so that
``math.nextafter`` can step across it. None of them is the rung's d20 target: the plan states
T-R4's target as the word "checkpoint" and no number, so the four real thresholds are Huy's, they
live in ``experiments/T1/T-R4/d20_target.json``, and :meth:`D20Target.from_mapping` refuses a null
rather than defaulting one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from scratch_llm.chat import CHAT_SPECIAL_TOKENS
from scratch_llm.eval.checkpoint_card import (
    CheckpointCard,
    D20Target,
    EvalProtocol,
    IncomparableCards,
    MissingMetric,
    compare,
    d20_gate,
    load_cards,
    save_cards,
    split_id,
)
from scratch_llm.eval.vibe import (
    VIBE_PROMPTS,
    VIBE_SET_ID,
    VibeDecode,
    VibeMismatch,
    VibeOutput,
    VibeRun,
    diff_runs,
    prompt_seed,
    prompts_fingerprint,
    run_vibe_evals,
    runs_identical,
    vibe_metrics,
)
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.tokenizer import Tokenizer, train_bpe
from scratch_llm.train import build_model_from_checkpoint, save_checkpoint

_CORPUS = "the cat sat on the mat. the dog ran! hello world, ok? banana france blue\n" * 20

_BOX = (
    "GPU box only — run it there: infra/rent.sh sync-up <user@host> && ssh <user@host> "
    "&& bash ladders/experiments/T1/T-R4/run.sh"
)


def _chat_tokenizer(tmp_path: Path) -> Tokenizer:
    """A small BPE with CHAT_SPECIAL_TOKENS threaded through train_bpe AND the constructor."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=300, special_tokens=CHAT_SPECIAL_TOKENS)
    return Tokenizer(vocab, merges, special_tokens=CHAT_SPECIAL_TOKENS)


def _tiny_model(vocab_size: int, seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    return TransformerLM(
        ModelConfig(vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=2, context_length=192)
    )


def _protocol(**over: object) -> EvalProtocol:
    """A complete protocol in all three scopes; ``over`` perturbs exactly one field per test."""
    base = {
        "core_tasks": ("arc_easy", "winograd"),
        "core_limit": 20,
        "core_bundle": "bundle-v1",
        "val_split_id": split_id(b"held-out text", "val"),
        "val_num_bytes": 13,
        "vibe_set_id": VIBE_SET_ID,
        "vibe_prompts_hash": prompts_fingerprint(VIBE_PROMPTS),
        "vibe_decode_id": VibeDecode().id,
    }
    base.update(over)
    return EvalProtocol(**base)  # type: ignore[arg-type]


def _card(stage: str, *, protocol: EvalProtocol | None = None, **metrics: float) -> CheckpointCard:
    defaults: dict[str, dict[str, float]] = {
        "pretrain": {"val_bpb": 1.0, "core": 0.5},
        "midtrain": {"val_bpb": 1.0, "core": 0.5},
        "sft": {"val_bpb": 1.0, "core": 0.5, "vibe_eot_rate": 1.0, "vibe_deterministic": 1.0},
        "reference": {"core": 0.5},
    }
    values = dict(defaults[stage])
    values.update(metrics)
    return CheckpointCard(
        checkpoint_id=f"{stage}@test",
        stage=stage,  # type: ignore[arg-type]
        step=10,
        n_params=1000,
        tokens_seen=10,
        protocol=protocol if protocol is not None else _protocol(),
        metrics=values,
    )


# ---------------------------------------------------------------------------------------------
# 1. Refusal — a comparison across two protocols measures the protocol
# ---------------------------------------------------------------------------------------------


def test_compare_refuses_different_vibe_prompt_set() -> None:
    a = _card("pretrain")
    b = _card("sft", protocol=_protocol(vibe_prompts_hash="deadbeefdeadbeef"))
    with pytest.raises(IncomparableCards, match="vibe_prompts_hash"):
        compare(a, b, scope="vibe")


def test_compare_refuses_different_decode_config() -> None:
    a = _card("pretrain")
    b = _card("sft", protocol=_protocol(vibe_decode_id=VibeDecode(temperature=0.7).id))
    with pytest.raises(IncomparableCards, match="vibe_decode_id"):
        compare(a, b, scope="vibe")


def test_compare_refuses_different_val_split() -> None:
    a = _card("pretrain")
    b = _card("sft", protocol=_protocol(val_split_id=split_id(b"other text", "val")))
    with pytest.raises(IncomparableCards, match="val_split_id"):
        compare(a, b, scope="val")


def test_compare_refuses_a_scope_the_card_never_evaluated() -> None:
    """The floor card has a CORE number and no held-out split of ours — say so, don't invent one."""
    ours = _card("pretrain")
    floor = _card(
        "reference",
        protocol=EvalProtocol(
            core_tasks=("arc_easy", "winograd"), core_limit=20, core_bundle="bundle-v1"
        ),
    )
    with pytest.raises(IncomparableCards, match="carries no val evaluation"):
        compare(ours, floor, scope="val")


def test_floor_card_compares_on_core_scope() -> None:
    """Same tasks, same limit, same bundle ⇒ the CORE comparison is legal even across tokenizers."""
    ours = _card("pretrain", core=0.30)
    floor = _card(
        "reference",
        protocol=EvalProtocol(
            core_tasks=("arc_easy", "winograd"), core_limit=20, core_bundle="bundle-v1"
        ),
        core=0.22,
    )
    deltas = compare(ours, floor, scope="core")
    assert deltas["core"][0] == pytest.approx(0.30)
    assert deltas["core"][1] == pytest.approx(0.22)
    assert deltas["core"][2] == pytest.approx(-0.08)


def test_matching_protocols_compare_cleanly() -> None:
    a = _card("pretrain", core=0.20, val_bpb=1.20)
    b = _card("sft", core=0.19, val_bpb=1.21)
    assert compare(a, b, scope="core")["core"][2] == pytest.approx(-0.01)
    assert compare(a, b, scope="val")["val_bpb"][2] == pytest.approx(0.01)


# ---------------------------------------------------------------------------------------------
# 2. Determinism — the same checkpoint twice is the same tokens
# ---------------------------------------------------------------------------------------------


def test_same_checkpoint_twice_is_identical_greedy(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    first = run_vibe_evals(model, tok)
    second = run_vibe_evals(model, tok)
    assert runs_identical(first, second)
    assert diff_runs(first, second) == ()


def test_same_checkpoint_twice_is_identical_at_temperature(tmp_path: Path) -> None:
    """The claim that matters: seeded sampling, not greedy collapse, reproduces exactly."""
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    decode = VibeDecode(temperature=0.9, top_p=0.95, max_tokens=24)
    first = run_vibe_evals(model, tok, decode=decode)
    second = run_vibe_evals(model, tok, decode=decode)
    assert runs_identical(first, second)
    # ...and it is genuinely sampling, not greedy in disguise: another base seed moves a probe.
    # Note the comparison has to be made by hand — diff_runs refuses across seeds, because the
    # seed is part of the decode config and a delta measured across two seeds is sampler noise.
    other = run_vibe_evals(
        model, tok, decode=VibeDecode(temperature=0.9, top_p=0.95, max_tokens=24, seed=1)
    )
    with pytest.raises(VibeMismatch, match="decode config differs"):
        diff_runs(first, other)
    left, right = first.by_prompt(), other.by_prompt()
    assert any(left[pid].token_ids != right[pid].token_ids for pid in left)


def test_vibe_is_invariant_to_prompt_order(tmp_path: Path) -> None:
    """Per-prompt seeds, not one RNG stream: prompt #7 does not depend on prompts #0..#6."""
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    decode = VibeDecode(temperature=0.8, max_tokens=16)
    forward = run_vibe_evals(model, tok, decode=decode)
    subset = list(reversed(VIBE_PROMPTS))[:4]
    reversed_run = run_vibe_evals(model, tok, decode=decode, prompts=subset)
    by_prompt = forward.by_prompt()
    for out in reversed_run.outputs:
        assert out.token_ids == by_prompt[out.prompt_id].token_ids


def test_reloaded_checkpoint_reproduces_its_vibe_run(tmp_path: Path) -> None:
    """Determinism across a checkpoint round-trip — the shape the CLI actually runs in."""
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    decode = VibeDecode(temperature=0.7, max_tokens=16)
    before = run_vibe_evals(model, tok, decode=decode)

    ckpt = tmp_path / "sft.pt"
    save_checkpoint(model, None, 10, ckpt)
    reloaded, step = build_model_from_checkpoint(ckpt)
    assert step == 10
    after = run_vibe_evals(reloaded, tok, decode=decode)
    assert runs_identical(before, after)


def test_prompt_seed_is_stable_and_prompt_specific() -> None:
    assert prompt_seed(0, "greet") == prompt_seed(0, "greet")  # not Python's salted hash
    assert prompt_seed(0, "greet") != prompt_seed(0, "identity")
    assert prompt_seed(0, "greet") != prompt_seed(1, "greet")


def test_vibe_refuses_a_different_prompt_set(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    full = run_vibe_evals(model, tok)
    subset = run_vibe_evals(model, tok, prompts=VIBE_PROMPTS[:3])
    with pytest.raises(VibeMismatch, match="prompt set differs"):
        diff_runs(full, subset)


def test_vibe_refuses_a_different_decode(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    a = run_vibe_evals(model, tok, decode=VibeDecode(max_tokens=8))
    b = run_vibe_evals(model, tok, decode=VibeDecode(max_tokens=8, temperature=0.5))
    with pytest.raises(VibeMismatch, match="decode config differs"):
        diff_runs(a, b)


def test_raw_and_chat_templates_are_not_comparable(tmp_path: Path) -> None:
    """A base checkpoint is probed raw and an SFT checkpoint through the template; the two
    numbers are about different things, and the harness says so instead of subtracting them."""
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    raw = run_vibe_evals(model, tok, decode=VibeDecode(template="raw", max_tokens=8))
    chat = run_vibe_evals(model, tok, decode=VibeDecode(template="chat", max_tokens=8))
    assert "vibe_eot_rate" not in vibe_metrics(raw)  # no turn to end ⇒ absent, not 0.0
    assert "vibe_eot_rate" in vibe_metrics(chat)
    with pytest.raises(VibeMismatch, match="decode config differs"):
        diff_runs(raw, chat)


def test_chat_template_requires_chat_specials(tmp_path: Path) -> None:
    corpus = tmp_path / "plain.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=290, special_tokens=None)
    plain = Tokenizer(vocab, merges)
    model = _tiny_model(len(plain.vocab))
    with pytest.raises(VibeMismatch, match="CHAT_SPECIAL_TOKENS"):
        run_vibe_evals(model, plain, decode=VibeDecode(template="chat", max_tokens=4))


# ---------------------------------------------------------------------------------------------
# 3. No blank cells — a missing metric is an error
# ---------------------------------------------------------------------------------------------


def test_missing_required_metric_is_an_error_at_construction() -> None:
    with pytest.raises(MissingMetric, match="vibe_eot_rate"):
        CheckpointCard(
            checkpoint_id="sft@test",
            stage="sft",
            step=1,
            n_params=1000,
            tokens_seen=10,
            protocol=_protocol(),
            metrics={"val_bpb": 1.0, "core": 0.2, "vibe_deterministic": 1.0},
        )


def test_metric_lookup_raises_instead_of_returning_none() -> None:
    card = _card("pretrain")
    with pytest.raises(MissingMetric, match="gsm8k"):
        card.metric("gsm8k")


def test_non_finite_metric_is_an_error() -> None:
    with pytest.raises(MissingMetric, match="not a finite number"):
        _card("pretrain", core=float("nan"))


def test_tokens_seen_is_required_because_the_checkpoint_does_not_carry_it() -> None:
    with pytest.raises(ValueError, match="tokens_seen must be positive"):
        CheckpointCard(
            checkpoint_id="pretrain@test",
            stage="pretrain",
            step=1,
            n_params=1000,
            tokens_seen=0,
            protocol=_protocol(),
            metrics={"val_bpb": 1.0, "core": 0.2},
        )


def test_cards_round_trip_through_json(tmp_path: Path) -> None:
    card = _card("sft")
    out = tmp_path / "cards.json"
    save_cards([card], out)
    (restored,) = load_cards(out)
    assert restored.to_dict() == card.to_dict()


# ---------------------------------------------------------------------------------------------
# 4. Exactness — the gate at the boundary. Synthetic thresholds; see the module docstring.
# ---------------------------------------------------------------------------------------------

_TARGET = D20Target(min_core=0.5, max_val_bpb=1.0, min_tokens_seen=10, min_vibe_eot_rate=0.5)


def _pair(
    *,
    core: float = 0.5,
    val_bpb: float = 1.0,
    tokens_seen: int = 10,
    eot: float = 0.5,
    deterministic: float = 1.0,
    chat_protocol: EvalProtocol | None = None,
) -> tuple[CheckpointCard, CheckpointCard]:
    base = CheckpointCard(
        checkpoint_id="pretrain@d20",
        stage="pretrain",
        step=100,
        n_params=1000,
        tokens_seen=tokens_seen,
        protocol=_protocol(),
        metrics={"val_bpb": val_bpb, "core": core},
    )
    chat = CheckpointCard(
        checkpoint_id="sft@d20",
        stage="sft",
        step=110,
        n_params=1000,
        tokens_seen=tokens_seen,
        protocol=chat_protocol if chat_protocol is not None else _protocol(),
        metrics={
            "val_bpb": val_bpb,
            "core": core,
            "vibe_eot_rate": eot,
            "vibe_deterministic": deterministic,
        },
    )
    return base, chat


def test_gate_passes_exactly_at_the_boundary() -> None:
    """Every numeric clause sits ON its threshold: >= and <= pass at equality. No epsilon."""
    result = d20_gate(*_pair(), _TARGET)
    assert result.passed, result.to_table()
    assert [c.name for c in result.failures()] == []


@pytest.mark.parametrize(
    ("kwarg", "clause", "direction"),
    [
        ("core", "core", -1.0),  # one ulp below min_core
        ("val_bpb", "val_bpb", +1.0),  # one ulp above max_val_bpb
        ("eot", "vibe_eot_rate", -1.0),
    ],
)
def test_gate_fails_one_ulp_beyond_the_boundary(kwarg: str, clause: str, direction: float) -> None:
    at = {"core": 0.5, "val_bpb": 1.0, "eot": 0.5}
    beyond = math.nextafter(at[kwarg], math.inf if direction > 0 else -math.inf)
    at[kwarg] = beyond
    result = d20_gate(
        *_pair(core=at["core"], val_bpb=at["val_bpb"], eot=at["eot"]),
        _TARGET,
    )
    assert not result.passed
    assert [c.name for c in result.failures()] == [clause]


def test_gate_fails_one_token_below_the_compute_budget() -> None:
    result = d20_gate(*_pair(tokens_seen=9), _TARGET)
    assert not result.passed
    assert [c.name for c in result.failures()] == ["tokens_seen"]


def test_gate_fails_when_the_vibe_rerun_was_not_identical() -> None:
    result = d20_gate(*_pair(deterministic=0.0), _TARGET)
    assert not result.passed
    assert [c.name for c in result.failures()] == ["vibe_deterministic"]


def test_gate_fails_when_the_two_cards_used_different_protocols() -> None:
    """Structural clause, no threshold involved: two cards scored differently are not a ladder."""
    result = d20_gate(
        *_pair(chat_protocol=_protocol(core_limit=50)),
        _TARGET,
    )
    assert not result.passed
    assert [c.name for c in result.failures()] == ["protocol_core"]


def test_gate_table_names_every_clause() -> None:
    table = d20_gate(*_pair(), _TARGET).to_table()
    for clause in (
        "protocol_core",
        "protocol_val",
        "same_architecture",
        "tokens_monotone",
        "vibe_deterministic",
        "core",
        "val_bpb",
        "tokens_seen",
        "vibe_eot_rate",
    ):
        assert clause in table


def test_target_refuses_null_fields_instead_of_defaulting_them() -> None:
    with pytest.raises(ValueError, match=r"min_core.*max_val_bpb|max_val_bpb"):
        D20Target.from_mapping({"min_tokens_seen": 1, "min_vibe_eot_rate": 1.0})


def test_target_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    path.write_text(
        json.dumps(
            {
                "min_core": 0.5,
                "max_val_bpb": 1.0,
                "min_tokens_seen": 10,
                "min_vibe_eot_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    assert D20Target.from_json(path) == _TARGET


def test_unfilled_target_file_names_the_file_and_the_fields(tmp_path: Path) -> None:
    path = tmp_path / "d20_target.json"
    path.write_text(json.dumps({k: None for k in ("min_core", "max_val_bpb")}), encoding="utf-8")
    with pytest.raises(ValueError, match=str(path)):
        D20Target.from_json(path)


# ---------------------------------------------------------------------------------------------
# Oracle — vibe_metrics against a hand-computed reference on a constructed run
# ---------------------------------------------------------------------------------------------


def test_vibe_metrics_match_a_hand_computed_reference() -> None:
    """Four probes: two terminated by eot, one truncated at the 8-token budget, one empty."""
    decode = VibeDecode(max_tokens=8)
    outputs = (
        VibeOutput("a", (5, 6, 9), "hi", True, 3),
        VibeOutput("b", (7, 9), "ok", True, 2),
        VibeOutput("c", (1, 2, 3, 4, 5, 6, 7, 8), "blah blah", False, 8),
        VibeOutput("d", (9,), "", True, 1),
    )
    run = VibeRun(VIBE_SET_ID, "hash", decode, outputs)
    metrics = vibe_metrics(run)
    assert metrics["vibe_eot_rate"] == pytest.approx(3 / 4)
    assert metrics["vibe_truncated_rate"] == pytest.approx(1 / 4)
    assert metrics["vibe_empty_text_rate"] == pytest.approx(1 / 4)
    assert metrics["vibe_mean_new_tokens"] == pytest.approx((3 + 2 + 8 + 1) / 4)


def test_vibe_run_json_round_trip(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    run = run_vibe_evals(model, tok, decode=VibeDecode(max_tokens=8))
    restored = VibeRun.from_dict(json.loads(json.dumps(run.to_dict())))
    assert restored.fingerprint == run.fingerprint
    assert runs_identical(restored, run)


# ---------------------------------------------------------------------------------------------
# GPU — skipped with the exact box command, never silently passed
# ---------------------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason=_BOX)
def test_vibe_is_deterministic_on_cuda(tmp_path: Path) -> None:
    """The rung's cards are produced on the 8xH100 box; determinism must hold there, where the
    kernels, not the reference path, decide the argmax."""
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab)).to("cuda")
    decode = VibeDecode(temperature=0.8, max_tokens=16)
    first = run_vibe_evals(model, tok, decode=decode, device="cuda")
    second = run_vibe_evals(model, tok, decode=decode, device="cuda")
    assert runs_identical(first, second)


# ---------------------------------------------------------------------------------------------
# The floor path — a reference card must be comparable with ours in the CORE scope, or the floor
# is not a floor. This is the failure that would otherwise surface only after the paid run.
# ---------------------------------------------------------------------------------------------


def _card_cli():
    """Import the bench CLI (bench/ is not a package), the way tests/test_f12_script.py does."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    try:
        import t1_speedrun_card
    finally:
        sys.path.pop(0)
    return t1_speedrun_card


def test_reference_card_from_report_is_comparable_with_a_rung_card(tmp_path: Path) -> None:
    """`--from-report` + the rung path must agree on the core fingerprint bit for bit — including
    when the operator lists the tasks in a different order, which canonicalisation absorbs."""
    cli = _card_cli()
    report = tmp_path / "nanochat_base_eval.json"
    report.write_text(json.dumps({"core": 0.2, "results": {}}), encoding="utf-8")
    out = tmp_path / "floor_card.json"
    cli.main(
        [
            "--from-report",
            str(report),
            "--reference-id",
            "nanochat-d20-base",
            "--tasks",
            "winograd,arc_easy",  # deliberately NOT manifest order
            "--limit",
            "20",
            "--bundle",
            "bundle-v1",
            "--n-params",
            "480400000",
            "--tokens-seen",
            "9600000000",
            "--out",
            str(out),
        ]
    )
    (floor,) = load_cards(out)
    assert floor.stage == "reference"
    assert floor.metric("core") == pytest.approx(0.2)

    # The rung path's own view of the same task list, taken through the same canonicaliser and
    # spelled in the other order — the fingerprints must still agree.
    rung_args = cli.build_parser().parse_args(["--tasks", "arc_easy,winograd"])
    ours = _card(
        "pretrain",
        protocol=_protocol(core_tasks=cli._canonical_tasks(rung_args), core_bundle="bundle-v1"),
    )
    assert compare(ours, floor, scope="core")["core"][1] == pytest.approx(0.2)


def test_reference_card_refuses_a_missing_key(tmp_path: Path) -> None:
    cli = _card_cli()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"summary": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="key path"):
        cli.main(
            [
                "--from-report",
                str(report),
                "--tasks",
                "arc_easy",
                "--n-params",
                "1",
                "--tokens-seen",
                "1",
                "--out",
                str(tmp_path / "x.json"),
            ]
        )


def test_reference_card_refuses_an_unknown_budget(tmp_path: Path) -> None:
    cli = _card_cli()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"core": 0.2}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a matched floor"):
        cli.main(
            [
                "--from-report",
                str(report),
                "--tasks",
                "arc_easy",
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
