"""The end-to-end speedrun spine — nanochat's "one script, whole loop" over our own components.

Chains the stages that turn a corpus into a model you can sample from:

    tokenizer (byte-level BPE) → pretrain (MuonAdamW) → eval (report card) → sample

or, with ``data_dir`` set (A1), pre-built shards replace the tokenizer stage:

    shards (data/shards.py: tokenizer.json + *.bin memmap) → pretrain → eval → sample

scaled by a single ``depth`` knob (nanochat's aspect-ratio scaling: ``d_model = 64·depth``,
``head_dim = 128``). Since A2 the run is a chain of stage functions with config-carrying,
optimizer-free checkpoints at each boundary (``work_dir``/``resume`` — the rental safety-net);
midtrain (A4) and SFT (A5) are declared slots that skip at 0 steps. RL reuses the same ``train``
loop + ``algos/`` (F2/F7). The **nano pre-flight**'s only job is to prove the whole pipeline
composes end-to-end in seconds on CPU **before** a $100 d20 rental (ADR-0018 §5, Phase 0).
Run it: ``python -m scratch_llm.speedrun --nano`` (or ``scripts/speedrun.sh``).

Model sizes (docs/FRONTIER_2026_ABLATIONS.md §2): ``--depth 20`` ⇒ d_model 1280 / 10 heads /
~561M params — the nanochat d20 headline; ``--nano`` ⇒ depth 4 in seconds.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scratch_llm.chat import CHAT_SPECIAL_TOKENS
from scratch_llm.data.shards import load_dataset_tokens, load_tokenizer
from scratch_llm.eval import ReportCard, build_report_card
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.tokenizer import Tokenizer, train_bpe
from scratch_llm.train import TrainConfig, build_model_from_checkpoint, save_checkpoint, train

# A tiny built-in corpus for the nano pre-flight (no external data needed). Repeated at runtime so
# the tokenizer has something to merge and the loader has enough tokens for a window.
_BUILTIN_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a language model learns to predict the next token from the previous ones. "
    "attention is all you need; the transformer reads the whole context at once. "
    "we own every layer from the byte to the reinforcement update. "
)


@dataclass
class SpeedrunConfig:
    depth: int = 20
    vocab_size: int = 65536  # nanochat's 2^16; the nano pre-flight overrides it small
    context_length: int = 1024
    train_steps: int = 1000
    batch_size: int = 32
    lr: float = 3e-3
    optimizer: str = "muon_adamw"  # F1 default; "adamw" for the A1 baseline
    amp_dtype: str | None = None  # "bf16" on the GPU (NOT with compile on sm120 — see train.py)
    compile: bool = False
    device: str = "cpu"
    seed: int = 0
    corpus_path: str | None = None  # None ⇒ the built-in nano corpus
    # A1 shard-backed path: a dataset dir built by data/shards.py (tokenizer.json + *.bin).
    # When set, corpus_path/vocab_size are ignored — the staged tokenizer defines the vocab axis.
    data_dir: str | None = None
    shard_glob: str = "*.bin"
    sample_tokens: int = 48
    sample_temperature: float = 0.8
    # A2 stage chaining. work_dir persists stage artifacts (tokenizer.json + <stage>.pt —
    # config-carrying, optimizer-free); resume=True rebuilds a stage from its artifact instead
    # of re-running it (the rental safety-net). midtrain/sft are the A4/A5 slots: 0 = skipped.
    work_dir: str | None = None
    resume: bool = False
    midtrain_steps: int = 0
    sft_steps: int = 0
    # A3: True ⇒ the tokenizer stage trains the BPE with CHAT_SPECIAL_TOKENS (each special
    # becomes a single id inside vocab_size — required by midtrain/SFT/chat, A4–A6).
    # False (default) is byte-identical to the pre-A3 path: no specials anywhere.
    chat: bool = False


@dataclass
class SpeedrunResult:
    config: SpeedrunConfig
    n_params: int
    n_tokens: int
    report_card: ReportCard
    sample: str
    stages: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"speedrun depth={self.config.depth} | params={self.n_params:,} | "
            f"tokens={self.n_tokens:,} | {self.seconds:.1f}s\n"
            f"stages: {' → '.join(self.stages)}\n{self.report_card.to_markdown()}\n"
            f"sample: {self.sample!r}"
        )


def model_config_for_depth(depth: int, vocab_size: int, context_length: int) -> ModelConfig:
    """nanochat aspect-ratio scaling: width grows with depth, head_dim pinned at 128 (or the whole
    width for tiny nano models). ``depth 20 → d_model 1280, 10 heads`` = the d20 headline."""
    d_model = 64 * depth
    n_heads = max(1, d_model // 128)
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=depth,
        n_heads=n_heads,
        context_length=context_length,
        tie_embeddings=False,  # untied (nanochat-style, F1 keeps Muon routing unambiguous)
    )


def _load_corpus(cfg: SpeedrunConfig) -> str:
    if cfg.corpus_path is not None:
        return Path(cfg.corpus_path).read_text(encoding="utf-8")
    # Repeat the built-in corpus so there are enough tokens for the window + batch sampling.
    target_chars = max(5000, cfg.context_length * 120)
    reps = target_chars // len(_BUILTIN_CORPUS) + 1
    return _BUILTIN_CORPUS * reps


def _train_tokenizer(
    text: str, vocab_size: int, special_tokens: list[str] | None = None
) -> Tokenizer:
    """Train a byte-level BPE on the corpus (train_bpe reads a file, so stage the text).

    ``special_tokens`` (A3) is threaded through BOTH train_bpe (ids inside vocab_size) and
    the Tokenizer constructor (encode maps each special to that single id) — the two halves
    of the A3 kill criterion.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        path = fh.name
    try:
        vocab, merges = train_bpe(path, vocab_size, special_tokens)
    finally:
        Path(path).unlink(missing_ok=True)
    return Tokenizer(vocab, merges, special_tokens)


# -----------------------------------------------------------------------------------------------
# A2 — the chained stage functions. Each stage is independently callable; stage boundaries
# persist config-carrying, optimizer-FREE checkpoints into cfg.work_dir (the pinned
# stage-transition policy: every stage builds a fresh optimizer with its own LR warmup —
# resuming Adam/Muon moments across an adamw↔muon_adamw switch is undefined; intra-run resume
# with optimizer state is train()'s separate checkpoint_every path). A4 midtrain / A5 SFT /
# A6 chat hang off this spine.
# -----------------------------------------------------------------------------------------------


def _work_path(cfg: SpeedrunConfig, name: str) -> Path | None:
    if cfg.work_dir is None:
        return None
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    return work / name


def stage_tokenizer(cfg: SpeedrunConfig) -> tuple[Tokenizer, np.ndarray, list[int], str]:
    """Stage 1 — the token axis: pre-built shards (A1), a work_dir-staged tokenizer (resume),
    or a corpus-trained byte BPE. Returns (tokenizer, tokens, prompt_ids, stage_name)."""
    if cfg.data_dir is not None:
        # A1 shard-backed pretrain: tokens + the tokenizer that produced them come from disk.
        tokenizer = load_tokenizer(cfg.data_dir)
        tokens = load_dataset_tokens(cfg.data_dir, cfg.shard_glob)
        prompt_ids = [int(t) for t in tokens[:8]]
        return tokenizer, tokens, prompt_ids, "shards"

    text = _load_corpus(cfg)
    tok_path = _work_path(cfg, "tokenizer.json")
    if cfg.resume and tok_path is not None and tok_path.exists():
        # BPE training is the expensive part at real vocab sizes — reload, re-encode only.
        tokenizer = Tokenizer.load(tok_path)
        name = "tokenizer[resumed]"
    else:
        specials = CHAT_SPECIAL_TOKENS if cfg.chat else None
        tokenizer = _train_tokenizer(text, cfg.vocab_size, specials)
        if tok_path is not None:
            tokenizer.save(tok_path)
        name = "tokenizer"
    tokens = np.asarray(tokenizer.encode(text), dtype=np.int64)
    return tokenizer, tokens, tokenizer.encode(text[:24]), name


def stage_pretrain(
    cfg: SpeedrunConfig, tokens: np.ndarray, vocab_size: int
) -> tuple[TransformerLM, str]:
    """Stage 2 — pretrain, or rebuild from the stage-boundary artifact when resume=True."""
    ckpt = _work_path(cfg, "pretrain.pt")
    if cfg.resume and ckpt is not None and ckpt.exists():
        model, _ = build_model_from_checkpoint(ckpt)
        model.to(cfg.device)
        return model, "pretrain[resumed]"

    model = TransformerLM(model_config_for_depth(cfg.depth, vocab_size, cfg.context_length))
    train(
        TrainConfig(
            max_steps=cfg.train_steps,
            batch_size=cfg.batch_size,
            context_length=cfg.context_length,
            max_lr=cfg.lr,
            warmup_steps=max(1, cfg.train_steps // 20),
            seed=cfg.seed,
            optimizer=cfg.optimizer,
            amp_dtype=cfg.amp_dtype,
            compile=cfg.compile,
            device=cfg.device,
        ),
        tokens,
        model,
    )
    if ckpt is not None:
        save_checkpoint(model, None, cfg.train_steps, ckpt)
    return model, "pretrain"


def stage_midtrain(cfg: SpeedrunConfig, model: TransformerLM) -> TransformerLM:
    """Stage 3 slot — chat-mix midtraining. 0 steps = skipped; A4 wires the body."""
    if cfg.midtrain_steps == 0:
        return model
    raise NotImplementedError(
        "midtrain is the A4 rung (docs/FRONTIER_2026_TASKSPEC.md §A4) — set midtrain_steps=0"
    )


def stage_sft(cfg: SpeedrunConfig, model: TransformerLM) -> TransformerLM:
    """Stage 4 slot — assistant-masked SFT. 0 steps = skipped; A5 wires the body."""
    if cfg.sft_steps == 0:
        return model
    raise NotImplementedError(
        "SFT is the A5 rung (docs/FRONTIER_2026_TASKSPEC.md §A5) — set sft_steps=0"
    )


def stage_eval(
    cfg: SpeedrunConfig, model: TransformerLM, tokenizer: Tokenizer, tokens: np.ndarray
) -> ReportCard:
    """Stage 5 — bits-per-byte on a tail slice. NOTE (nano): with the built-in repeated corpus
    this is in-sample — the pre-flight proves the metric computes; a real run supplies a held-out
    split via --corpus. num_bytes = the UTF-8 byte length the slice decodes to."""
    val_len = min(cfg.context_length * 4, tokens.size // 2)
    val = np.asarray(tokens[-val_len:], dtype=np.int64)  # int64 copy: shards memmap as uint16
    val_bytes = len(tokenizer.decode(val.tolist()).encode("utf-8"))
    return build_report_card(
        model,
        tokenizer,
        val_tokens=val,
        val_num_bytes=max(1, val_bytes),
        context_length=cfg.context_length,
        device=cfg.device,
    )


def stage_sample(
    cfg: SpeedrunConfig, model: TransformerLM, tokenizer: Tokenizer, prompt_ids: list[int]
) -> str:
    """Stage 6 — greedy-ish sample from the trained model (the talking artifact)."""
    budget = max(1, min(cfg.sample_tokens, cfg.context_length - len(prompt_ids) - 1))
    gen_ids = generate(
        model,
        prompt_ids,
        SamplingParams(temperature=cfg.sample_temperature, max_tokens=budget, seed=cfg.seed),
        device=cfg.device,
    )
    return tokenizer.decode(gen_ids)


def run_speedrun(cfg: SpeedrunConfig) -> SpeedrunResult:
    """Chain the stages: tokens → pretrain → [midtrain] → [sft] → eval → sample."""
    t0 = time.perf_counter()
    stages: list[str] = []

    tokenizer, tokens, prompt_ids, stage_name = stage_tokenizer(cfg)
    stages.append(stage_name)

    if tokens.size <= cfg.context_length + 1:
        raise ValueError(
            f"corpus encodes to {tokens.size} tokens, too short for context_length "
            f"{cfg.context_length}; supply a larger --corpus/--data-dir or a smaller --context."
        )

    # len(vocab) is the true id axis in both paths (train_bpe yields exactly vocab_size entries;
    # shard ids are defined by the staged tokenizer, never cfg.vocab_size).
    model, stage_name = stage_pretrain(cfg, tokens, len(tokenizer.vocab))
    stages.append(stage_name)

    model = stage_midtrain(cfg, model)
    if cfg.midtrain_steps:
        stages.append("midtrain")
    model = stage_sft(cfg, model)
    if cfg.sft_steps:
        stages.append("sft")

    card = stage_eval(cfg, model, tokenizer, tokens)
    stages.append("eval")

    sample = stage_sample(cfg, model, tokenizer, prompt_ids)
    stages.append("sample")

    return SpeedrunResult(
        config=cfg,
        n_params=sum(p.numel() for p in model.parameters()),
        n_tokens=int(tokens.size),
        report_card=card,
        sample=sample,
        stages=stages,
        seconds=time.perf_counter() - t0,
    )


def _nano_config() -> SpeedrunConfig:
    """Phase-0 pre-flight: the whole loop in ~a minute on CPU (protects the paid d20 run). Keep it
    light — its job is to prove the pipeline composes, not to train a good model; pass --device cuda
    (or a real --corpus + more --steps) once the plumbing is confirmed."""
    return SpeedrunConfig(
        depth=4,
        vocab_size=384,  # 256 bytes + 128 merges
        context_length=64,
        train_steps=60,
        batch_size=16,
        device="cpu",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Close-the-loop speedrun (tokenizer→pretrain→eval→sample)."
    )
    p.add_argument(
        "--nano", action="store_true", help="Phase-0 pre-flight (depth 4, seconds on CPU)."
    )
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--vocab", type=int, default=65536)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--optimizer", choices=["adamw", "muon_adamw"], default="muon_adamw")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (GPU).")
    p.add_argument(
        "--compile", action="store_true", help="torch.compile (not with --bf16 on sm120)."
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--corpus", default=None, help="Path to a text corpus (default: built-in nano corpus)."
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Shard dataset dir from data/shards.py (overrides --corpus/--vocab).",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Persist stage artifacts (tokenizer.json + pretrain.pt) here — the rental safety-net.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Rebuild finished stages from --work-dir artifacts instead of re-running them.",
    )
    args = p.parse_args()

    cfg = (
        _nano_config()
        if args.nano
        else SpeedrunConfig(
            depth=args.depth,
            vocab_size=args.vocab,
            context_length=args.context,
            train_steps=args.steps,
            batch_size=args.batch,
            lr=args.lr,
            optimizer=args.optimizer,
            amp_dtype="bf16" if args.bf16 else None,
            compile=args.compile,
            device=args.device,
            corpus_path=args.corpus,
            data_dir=args.data_dir,
            work_dir=args.work_dir,
            resume=args.resume,
        )
    )
    print(run_speedrun(cfg).summary())


if __name__ == "__main__":
    main()
