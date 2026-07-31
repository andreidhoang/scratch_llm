"""F12 corpus ablation — definition-of-done tests (CPU, toy corpora).

Covers the DoD invariants that need no GPU and no network: the pure bpb metric on fixture
bytes, the seeded held-out split, per-corpus tokenizer round-trips of a held-out doc,
loss-at-init ≈ ln V for each arm's own tokenizer, the held-out exclusion invariant (no
held-out doc survives into either arm's shards), and a full driver wiring smoke on toy
corpora with a fixture CORE task.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from scratch_llm.data.shards import load_dataset_tokens, load_tokenizer
from scratch_llm.eval.core_suite import TaskSpec
from scratch_llm.eval.corpus_ablation import (
    CorpusAblationArm,
    HeldOutSet,
    build_corpus_shard,
    encode_held_out,
    run_corpus_ablation,
    score_held_out_bpb,
    split_held_out,
)
from scratch_llm.eval.metrics import bits_per_byte
from scratch_llm.model import TransformerLM
from scratch_llm.speedrun import model_config_for_depth
from scratch_llm.train import TrainConfig
from scratch_llm.utils.seeding import seed_everything


def _sentence_docs(prefix: str, n: int) -> list[str]:
    """Distinct multi-sentence toy docs (enough volume for BPE merges and train windows)."""
    return [
        f"{prefix} document {i} tells a short story about language models and data. "
        f"the model reads every token of document {i} and learns to predict the next one. "
        f"good data makes the {prefix} model measurably better at this task."
        for i in range(n)
    ]


# ≥13 words each so the A0 13-gram gate catches any planted copy (the exclusion invariant).
_HELD_OUT_DOCS = (
    "held out evaluation document number one contains enough consecutive words to trip "
    "the thirteen gram decontamination gate whenever it leaks into training data",
    "held out evaluation document number two also carries a long run of plain words so "
    "that any verbatim copy inside a training shard is caught by the ngram gate",
)


def _corpora_with_planted_held_out() -> dict[CorpusAblationArm, list[str]]:
    """Both corpora carry the held-out docs verbatim — the gate must strip them per arm."""
    return {
        CorpusAblationArm.FINEWEB_EDU: _sentence_docs("fineweb", 24) + list(_HELD_OUT_DOCS),
        CorpusAblationArm.CLIMBMIX: _sentence_docs("climbmix", 24) + list(_HELD_OUT_DOCS),
    }


def _factories(corpora: dict[CorpusAblationArm, list[str]]):
    return {arm: (lambda docs=docs: [docs]) for arm, docs in corpora.items()}


class _UniformModel(torch.nn.Module):
    """All-equal logits over a V-vocab — the pure-metric fixture model (no training)."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ids.shape[0], ids.shape[1], self.vocab_size)


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------


def test_bpb_uniform_model_on_fixture_bytes() -> None:
    """A uniform model scores exactly log2(V) bits per token; bpb scales by bytes/token."""
    vocab, n = 1000, 40
    ids = list(range(n))
    num_bytes = len(b"hello world, these are fixture bytes.")
    result = bits_per_byte(_UniformModel(vocab), ids, num_bytes, context_length=64)
    assert result.n_tokens == n - 1  # every transition scored exactly once
    assert result.bits_per_byte == pytest.approx(math.log2(vocab) * (n - 1) / num_bytes)
    assert result.n_bytes == num_bytes


def test_split_held_out_excludes_and_is_deterministic() -> None:
    docs = _sentence_docs("pool", 10)
    train_a, held_a = split_held_out(docs, 3, seed=7)
    train_b, held_b = split_held_out(docs, 3, seed=7)
    assert len(train_a) == 7 and len(held_a.docs) == 3
    assert held_a == held_b and train_a == train_b
    assert not set(held_a.docs) & set(train_a)
    assert held_a.n_bytes == sum(len(d.encode("utf-8")) for d in held_a.docs)
    with pytest.raises(ValueError, match="n_held_out"):
        split_held_out(docs, 0)
    with pytest.raises(ValueError, match="n_held_out"):
        split_held_out(docs, len(docs))


# ---------------------------------------------------------------------------
# Per-arm tokenizer + shard staging
# ---------------------------------------------------------------------------


def test_per_corpus_tokenizer_round_trips_held_out_doc(tmp_path: Path) -> None:
    """Each arm's retrained BPE round-trips a held-out doc and encodes <|eot|> as one id."""
    held_out = HeldOutSet(docs=_HELD_OUT_DOCS)
    for arm in CorpusAblationArm:
        docs = _sentence_docs(arm.value, 20)
        out_dir = tmp_path / arm.value
        build_corpus_shard(
            arm,
            lambda docs=docs: [docs],
            out_dir,
            vocab_size=512,
            max_train_bytes=1 << 20,
            held_out=held_out,
        )
        tokenizer = load_tokenizer(out_dir)
        for doc in held_out.docs:
            assert tokenizer.decode(tokenizer.encode(doc)) == doc
        ids = encode_held_out(tokenizer, held_out)
        assert len(ids) > len(held_out.docs)  # content tokens plus one eot per doc


def test_build_corpus_shard_strips_planted_held_out_docs(tmp_path: Path) -> None:
    """The held-out exclusion invariant: a planted held-out doc never reaches the shards."""
    held_out = HeldOutSet(docs=_HELD_OUT_DOCS)
    corpora = _corpora_with_planted_held_out()
    for arm, docs in corpora.items():
        out_dir = tmp_path / arm.value
        _, stats = build_corpus_shard(
            arm,
            lambda docs=docs: [docs],
            out_dir,
            vocab_size=512,
            max_train_bytes=1 << 20,
            held_out=held_out,
        )
        assert stats.n_docs_filtered == len(_HELD_OUT_DOCS)
        tokenizer = load_tokenizer(out_dir)
        shard_text = tokenizer.decode([int(t) for t in load_dataset_tokens(out_dir)])
        for doc in held_out.docs:
            assert doc not in shard_text


# ---------------------------------------------------------------------------
# Loss at init ≈ ln V, per arm tokenizer
# ---------------------------------------------------------------------------


def test_loss_at_init_approx_log_vocab_per_arm(tmp_path: Path) -> None:
    held_out = HeldOutSet(docs=_HELD_OUT_DOCS)
    for arm in CorpusAblationArm:
        docs = _sentence_docs(arm.value, 20)
        build_corpus_shard(
            arm,
            lambda docs=docs: [docs],
            tmp_path / arm.value,
            vocab_size=512,
            max_train_bytes=1 << 20,
            held_out=held_out,
        )
        tokenizer = load_tokenizer(tmp_path / arm.value)
        vocab = len(tokenizer.vocab)
        seed_everything(0)
        model = TransformerLM(model_config_for_depth(1, vocab, context_length=32))
        bpb = score_held_out_bpb(model, tokenizer, held_out, context_length=32)
        # ≈uniform init logits ⇒ nats/token ≈ ln V (bpb's own numerator carries it).
        assert bpb.nats_per_token == pytest.approx(math.log(vocab), abs=0.2)


# ---------------------------------------------------------------------------
# Driver wiring smoke
# ---------------------------------------------------------------------------


def _toy_core_specs() -> tuple[TaskSpec, ...]:
    examples = [
        {"query": "which animal barks", "choices": ["the dog", "the cat"], "gold": 0},
        {"query": "which animal meows", "choices": ["the dog", "the cat"], "gold": 1},
    ]
    return (
        TaskSpec(
            name="toy_mc",
            task_type="multiple_choice",
            loader=lambda: examples,
            random_baseline=0.5,
        ),
    )


def test_run_corpus_ablation_smoke(tmp_path: Path) -> None:
    """Both arms stage, train (tiny steps), and score the SAME held-out bytes; the verdict
    and per-arm overlap rates are populated; the result serializes to JSON."""
    held_out = HeldOutSet(docs=_HELD_OUT_DOCS)
    corpora = _factories(_corpora_with_planted_held_out())
    train_cfg = TrainConfig(
        max_steps=2,
        batch_size=2,
        context_length=16,
        max_lr=1e-3,
        warmup_steps=1,
        eval_every=1,
        eval_batches=2,
        optimizer="muon_adamw",
        device="cpu",
        seed=0,
    )
    result = run_corpus_ablation(
        corpora,
        held_out,
        tmp_path / "f12",
        train_cfg,
        depth=1,
        vocab_size=512,
        max_train_bytes=1 << 20,
        core_specs=_toy_core_specs(),
    )

    for arm_result in (result.baseline, result.challenger):
        assert math.isfinite(arm_result.bpb.bits_per_byte)
        assert arm_result.bpb.n_bytes == held_out.n_bytes  # shared raw-byte denominator
        assert arm_result.init_val_ce == pytest.approx(math.log(arm_result.vocab_size), abs=0.2)
        assert (
            arm_result.overlap_rate is not None and arm_result.overlap_rate > 0.0
        )  # planted held-out docs dropped + logged
        assert arm_result.val_curve, "eval_every>0 must yield a val curve"
        assert arm_result.core is not None and math.isfinite(arm_result.core.core)
        assert arm_result.n_tokens == 2 * 2 * 16  # same TrainConfig ⇒ iso-FLOP by construction

    assert result.held_out_n_docs == len(_HELD_OUT_DOCS)
    assert result.bpb_delta == pytest.approx(
        result.challenger.bpb.bits_per_byte - result.baseline.bpb.bits_per_byte
    )
    assert result.verdict in {"climbmix_wins", "keep_fineweb_edu"}
    json.dumps(result.to_dict())  # ledger-serializable

    # Held-out exclusion through the FULL driver path (not just build_corpus_shard).
    for arm in CorpusAblationArm:
        arm_dir = tmp_path / "f12" / arm.value
        tokenizer = load_tokenizer(arm_dir)
        shard_text = tokenizer.decode([int(t) for t in load_dataset_tokens(arm_dir)])
        for doc in _HELD_OUT_DOCS:
            assert doc not in shard_text


def test_run_corpus_ablation_requires_both_arms_and_val_curve(tmp_path: Path) -> None:
    held_out = HeldOutSet(docs=_HELD_OUT_DOCS)
    train_cfg = TrainConfig(max_steps=1, batch_size=1, context_length=16, eval_every=1)
    with pytest.raises(ValueError, match="missing arm"):
        run_corpus_ablation(
            {CorpusAblationArm.FINEWEB_EDU: lambda: [_sentence_docs("a", 8)]},
            held_out,
            tmp_path / "a",
            train_cfg,
            depth=1,
            vocab_size=512,
        )
    with pytest.raises(ValueError, match="eval_every"):
        run_corpus_ablation(
            _factories(_corpora_with_planted_held_out()),
            held_out,
            tmp_path / "b",
            TrainConfig(max_steps=1, batch_size=1, context_length=16, eval_every=0),
            depth=1,
            vocab_size=512,
        )
