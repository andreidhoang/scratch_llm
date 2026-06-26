"""End-to-end L1 integration: tokenizer → model → training → sampling.

Unit tests prove each piece; this proves they *compose* into a working substrate — the
same path the rollout engine will drive (encode a prompt, run the policy, decode samples).
"""

from pathlib import Path

import numpy as np

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.tokenizer import Tokenizer, train_bpe
from scratch_llm.train import TrainConfig, train
from scratch_llm.utils.seeding import seed_everything


def test_l1_substrate_composes_end_to_end(tmp_path: Path) -> None:
    seed_everything(0)
    # A small, structured corpus so a tiny LM has something learnable.
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("the cat sat on the mat . the dog ran in the park . " * 200), encoding="utf-8"
    )
    special = "<|endoftext|>"

    # 1) train the tokenizer, then encode the corpus to a flat token array
    vocab, merges = train_bpe(corpus, vocab_size=300, special_tokens=[special])
    tok = Tokenizer(vocab, merges, special_tokens=[special])
    text = corpus.read_text(encoding="utf-8")
    ids = np.array(tok.encode(text), dtype=np.int64)
    assert ids.max() < len(vocab)

    # round-trip survives the tokenizer
    assert tok.decode(tok.encode("the cat sat")) == "the cat sat"

    # 2) train a tiny LM on the tokenized corpus
    model = TransformerLM(
        ModelConfig(vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4, context_length=32)
    )
    cfg = TrainConfig(
        max_steps=200, batch_size=16, context_length=24, max_lr=3e-3, warmup_steps=20, seed=0
    )
    history = train(cfg, ids, model)
    assert history[-1][1] < history[0][1], "training did not reduce loss end-to-end"

    # 3) sample a continuation and decode it back to text
    prompt = tok.encode("the cat")
    out_ids = generate(model, prompt, SamplingParams(temperature=0.0, max_tokens=10))
    assert len(out_ids) == 10
    completion = tok.decode(out_ids)
    assert isinstance(completion, str) and len(completion) > 0
