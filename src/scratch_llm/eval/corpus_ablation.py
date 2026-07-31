"""F12 corpus ablation — FineWeb-EDU vs ClimbMix at iso-FLOP, each with its OWN retrained BPE.

The d20 data decision (`docs/FRONTIER_2026_TASKSPEC.md` §F12): nanochat's switch
FineWeb-EDU-100B → ClimbMix-400B was its single biggest speedrun win (−27% wall-clock,
val_bpb 0.7465 → 0.7185) — but confounded with a d26→d24 depth change. This harness is the
novel **iso-FLOP** measurement at ~35.8M params: two arms, identical model
(``model_config_for_depth(6)``), token budget, seed, and Muon+AdamW recipe; only the corpus —
and the tokenizer retrained ON that corpus — differs.

Design invariants:
- **Iso-FLOP by construction:** both arms share N (same depth/vocab, same seeded init) and D
  (same ``TrainConfig`` step/batch/context budget), so C = 6ND is held constant without
  bookkeeping; the driver re-checks the token counts match before returning a verdict.
- **Tokenizer per arm, same byte budget:** each arm's BPE is trained on a
  ``max_train_bytes``-capped sample of its OWN corpus (comparing bpb with a foreign
  tokenizer conflates corpus with tokenizer fit).
- **Shared raw-byte eval:** both arms score the SAME held-out documents —
  ``bits_per_byte`` normalizes per UTF-8 byte, so the comparison is fair across tokenizers.
  The held-out docs are kept out of BOTH arms' training shards AND BPE samples via the A0
  13-gram gate (``data/decontaminate.py``), with the overlap (drop) rate logged per arm.
- **CORE is secondary:** its single-seed noise floor (±0.008–0.016) is far above the expected
  bpb delta, so the verdict reads bpb only; CORE rides along when ``core_specs`` is passed.

Pre-registered falsifier: ClimbMix bpb < FineWeb-EDU bpb at iso-FLOP (predicted Δ ≈ −0.010 to
−0.030). KILL: ClimbMix ≥ FWE ⇒ keep the banked FineWeb-EDU corpus; do not stage ClimbMix.

Interview question this answers: "why can a corpus swap beat an optimizer swap at fixed
compute — and why does comparing bpb require retraining the tokenizer on each corpus?"
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from scratch_llm.data.decontaminate import (
    build_eval_ngrams,
    decontam_doc_filter,
    default_eval_texts,
)
from scratch_llm.data.shards import (
    CLIMBMIX_DATASET,
    CLIMBMIX_SMALL_SUBDIR,
    CLIMBMIX_TOKENS_COLUMN,
    DEFAULT_BPE_TRAIN_BYTES,
    DEFAULT_TOKENS_PER_SHARD,
    DOC_SEPARATOR,
    FINEWEB_EDU_DATASET,
    FINEWEB_EDU_SAMPLE_10BT,
    ShardMeta,
    StreamStats,
    build_dataset_streaming,
    build_gpt2_detokenizer,
    iter_parquet_doc_batches,
    load_dataset_tokens,
    load_tokenizer,
)
from scratch_llm.eval.core_suite import CoreSuiteResult, TaskSpec, evaluate_core_suite
from scratch_llm.eval.metrics import BpbResult, bits_per_byte
from scratch_llm.model import TransformerLM, cross_entropy
from scratch_llm.scaling.isoflop import compute_from_params_tokens
from scratch_llm.speedrun import model_config_for_depth
from scratch_llm.tokenizer import Tokenizer
from scratch_llm.train import TrainConfig, train
from scratch_llm.utils.seeding import seed_everything


class CorpusAblationArm(Enum):
    """The two F12 arms — identical recipe; only the corpus + retrained tokenizer differ."""

    FINEWEB_EDU = "fineweb_edu"
    CLIMBMIX = "climbmix"


@dataclass(frozen=True)
class CorpusSource:
    """How to stage one arm's corpus from the HF hub parquet bulk path (shards.py).

    ``gpt2_detokenize=True`` marks a corpus whose ``text_column`` holds GPT-2 token ids
    instead of raw text — ClimbMix's verified layout (see the CLIMBMIX_* constants in
    shards.py); its docs are detokenized via :func:`build_gpt2_detokenizer`.
    """

    repo_id: str
    subdir: str
    text_column: str = "text"
    gpt2_detokenize: bool = False


CORPUS_SOURCES: dict[CorpusAblationArm, CorpusSource] = {
    CorpusAblationArm.FINEWEB_EDU: CorpusSource(FINEWEB_EDU_DATASET, FINEWEB_EDU_SAMPLE_10BT),
    CorpusAblationArm.CLIMBMIX: CorpusSource(
        CLIMBMIX_DATASET,
        CLIMBMIX_SMALL_SUBDIR,
        text_column=CLIMBMIX_TOKENS_COLUMN,
        gpt2_detokenize=True,
    ),
}


def hub_doc_batches(
    source: CorpusSource,
    parquet_paths: Sequence[str | Path],
    cache_dir: str | Path,
    *,
    batch_size: int = 1024,
) -> Callable[[], Iterable[Sequence[str]]]:
    """A zero-arg doc-batch factory over downloaded hub parquet (the real-run input seam).

    ``cache_dir`` is where the GPT-2 encoder.json lands when the source needs detokenizing.
    """
    decode = build_gpt2_detokenizer(cache_dir) if source.gpt2_detokenize else None
    return lambda: iter_parquet_doc_batches(
        parquet_paths, text_column=source.text_column, batch_size=batch_size, decode=decode
    )


# ---------------------------------------------------------------------------------------------
# The shared held-out set — fixed raw bytes, held out of BOTH arms, A0-decontaminated
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldOutSet:
    """The shared eval set: raw held-out document texts (bytes are the bpb denominator)."""

    docs: tuple[str, ...]

    @property
    def n_bytes(self) -> int:
        return sum(len(d.encode("utf-8")) for d in self.docs)


def split_held_out(
    docs: Sequence[str], n_held_out: int, *, seed: int = 1234
) -> tuple[list[str], HeldOutSet]:
    """Seeded doc-level split for list-like corpora: the held-out docs are REMOVED from the
    returned train list (exact exclusion by construction — the 13-gram gate in
    :func:`build_corpus_shard` additionally catches near-duplicates). Held-out order is the
    sorted sample order, so the byte total is deterministic.
    """
    if not 0 < n_held_out < len(docs):
        raise ValueError(
            f"n_held_out={n_held_out} must be in (0, {len(docs)}) for {len(docs)} docs"
        )
    rng = random.Random(seed)
    held_idx = sorted(rng.sample(range(len(docs)), n_held_out))
    held_set = set(held_idx)
    train_docs = [doc for i, doc in enumerate(docs) if i not in held_set]
    return train_docs, HeldOutSet(docs=tuple(docs[i] for i in held_idx))


def build_corpus_shard(
    arm: CorpusAblationArm,
    doc_batches: Callable[[], Iterable[Sequence[str]]],
    out_dir: str | Path,
    *,
    vocab_size: int,
    max_train_bytes: int = DEFAULT_BPE_TRAIN_BYTES,
    target_tokens: int | None = None,
    held_out: HeldOutSet | None = None,
    decontaminate: bool = True,
    extra_eval_paths: Sequence[str | Path] = (),
    tokens_per_shard: int = DEFAULT_TOKENS_PER_SHARD,
    num_workers: int = 0,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[ShardMeta], StreamStats]:
    """Stage one arm: retrain the BPE on the corpus's byte-capped sample, then stream-tokenize
    to shards — with the held-out docs kept OUT of both (the A0 13-gram gate).

    Thin wrapper over :func:`scratch_llm.data.shards.build_dataset_streaming` that builds the
    guard set from the held-out docs PLUS the default eval texts (so eval text never shapes
    the merges either). The per-arm drop rate lands in ``StreamStats.n_docs_filtered`` — the
    DoD's logged overlap rate (``run_corpus_ablation`` surfaces it as ``overlap_rate``).
    """
    doc_filter = None
    if decontaminate:
        guard_texts = list(held_out.docs) if held_out is not None else []
        guard_texts += default_eval_texts(extra_eval_paths)
        doc_filter = decontam_doc_filter(build_eval_ngrams(guard_texts))
    return build_dataset_streaming(
        doc_batches,
        out_dir,
        vocab_size,
        max_train_bytes=max_train_bytes,
        tokens_per_shard=tokens_per_shard,
        target_tokens=target_tokens,
        num_workers=num_workers,
        doc_filter=doc_filter,
        progress=progress,
    )


# ---------------------------------------------------------------------------------------------
# Held-out scoring — each arm scores the same raw bytes with its OWN tokenizer
# ---------------------------------------------------------------------------------------------


def encode_held_out(tokenizer: Tokenizer, held_out: HeldOutSet) -> list[int]:
    """The held-out docs as one token stream (``<|eot|>`` after every doc, matching the shard
    format), encoded with the ARM's tokenizer."""
    eot_ids = tokenizer.encode(DOC_SEPARATOR)
    if len(eot_ids) != 1:
        raise RuntimeError(f"{DOC_SEPARATOR!r} must encode to exactly one id, got {eot_ids}")
    ids: list[int] = []
    for doc in held_out.docs:
        ids.extend(tokenizer.encode(doc))
        ids.append(eot_ids[0])
    return ids


def score_held_out_bpb(
    model: TransformerLM,
    tokenizer: Tokenizer,
    held_out: HeldOutSet,
    *,
    context_length: int,
    device: str = "cpu",
) -> BpbResult:
    """val_bpb on the SHARED held-out bytes: NLL of the arm's token stream over the raw UTF-8
    byte count — the byte denominator is what makes the cross-tokenizer comparison fair."""
    ids = encode_held_out(tokenizer, held_out)
    return bits_per_byte(model, ids, held_out.n_bytes, context_length=context_length, device=device)


@torch.no_grad()
def loss_at_init(
    model: TransformerLM,
    token_ids: Sequence[int],
    *,
    context_length: int,
    device: str = "cpu",
) -> float:
    """Mean CE of an untrained model on the first context window — the DoD sanity check
    (≈ ln(vocab_size) for ≈uniform init logits)."""
    ids = torch.as_tensor(list(token_ids[: context_length + 1]), dtype=torch.long)
    if ids.numel() < 2:
        raise ValueError("need at least 2 tokens to score a transition")
    logits = model(ids[:-1].unsqueeze(0).to(device))
    return float(cross_entropy(logits, ids[1:].unsqueeze(0).to(device)).item())


# ---------------------------------------------------------------------------------------------
# The A/B driver — same iso-FLOP curve discipline as eval/optimizer_race.py
# ---------------------------------------------------------------------------------------------


@dataclass
class CorpusArmResult:
    """One arm of the ablation: its tokenizer/shard accounting, curves, and held-out scores."""

    arm: CorpusAblationArm
    n_params: int
    n_tokens: int  # D — tokens trained (max_steps·batch·ctx; must match across arms)
    n_shard_tokens: int
    vocab_size: int  # the arm's OWN retrained vocab (== requested unless merges exhausted)
    init_val_ce: float  # ≈ ln(vocab_size) — the DoD loss-at-init check
    bpb: BpbResult  # PRIMARY: held-out raw bytes, arm's own tokenizer
    core: CoreSuiteResult | None  # SECONDARY (±0.008–0.016 single-seed noise; bpb decides)
    # Decontam drop fraction of the arm's corpus — logged per arm (DoD). None ⇒ the arm
    # adopted prebuilt shards without a build_stats.json (a .bin cannot be re-filtered after
    # the fact, so the rate is honestly unknown — never fabricated).
    overlap_rate: float | None
    train_history: list[tuple[int, float]]
    val_curve: list[tuple[float, float]]  # (tokens_seen, val_ce) on fixed windows
    wall_seconds: float


def arm_result_to_dict(r: CorpusArmResult) -> dict[str, object]:
    """A JSON-safe dump of one arm — the ``arm_hook`` incremental-persistence payload."""
    return {
        "arm": r.arm.value,
        "n_params": r.n_params,
        "n_tokens": r.n_tokens,
        "n_shard_tokens": r.n_shard_tokens,
        "vocab_size": r.vocab_size,
        "init_val_ce": r.init_val_ce,
        "val_bpb": r.bpb.bits_per_byte,
        "bpb_nats_per_token": r.bpb.nats_per_token,
        "bpb_n_tokens": r.bpb.n_tokens,
        "bpb_n_bytes": r.bpb.n_bytes,
        "core": r.core.to_dict() if r.core is not None else None,
        "overlap_rate": r.overlap_rate,
        "val_curve": r.val_curve,
        "train_history": r.train_history,
        "wall_seconds": r.wall_seconds,
    }


# Build-time staging accounting, persisted next to the shards so an arm adopted via
# ``prebuilt`` (banked shards / a resumed run) still logs its decontamination overlap rate.
_BUILD_STATS_FILE = "build_stats.json"


def _write_build_stats(shard_dir: Path, stats: StreamStats) -> None:
    payload = json.dumps(asdict(stats), indent=2)
    (shard_dir / _BUILD_STATS_FILE).write_text(payload, encoding="utf-8")


def _read_build_stats(shard_dir: Path) -> StreamStats | None:
    path = shard_dir / _BUILD_STATS_FILE
    if not path.exists():
        return None
    return StreamStats(**json.loads(path.read_text(encoding="utf-8")))


def _run_arm(
    arm: CorpusAblationArm,
    doc_batches: Callable[[], Iterable[Sequence[str]]] | None,
    held_out: HeldOutSet,
    work_dir: Path,
    train_cfg: TrainConfig,
    *,
    depth: int,
    vocab_size: int,
    max_train_bytes: int,
    target_tokens: int | None,
    decontaminate: bool,
    extra_eval_paths: Sequence[str | Path],
    core_specs: Sequence[TaskSpec] | None,
    core_limit: int | None,
    num_workers: int,
    prebuilt_dir: str | Path | None,
    progress: Callable[[str], None] | None,
) -> CorpusArmResult:
    """Stage → train → score one arm. Seeded BEFORE model construction, so both arms share the
    initial weights when their dims match (both vocab 32768 at the F12 operating point).

    ``prebuilt_dir`` adopts an ALREADY-STAGED shard dir (banked shards, or a previous run's
    ``work_dir/<arm>/`` — the resume path): staging is skipped and the build-time accounting
    comes from its ``build_stats.json`` when present. The held-out decontamination gate can
    only run at build time, so adopting foreign shards built without it is the caller's
    documented responsibility.
    """
    t0 = time.perf_counter()
    stats: StreamStats | None
    if prebuilt_dir is not None:
        shard_dir = Path(prebuilt_dir)
        stats = _read_build_stats(shard_dir)
        if stats is None and progress is not None:
            progress(
                f"{arm.value}: adopted prebuilt shards {shard_dir} (no {_BUILD_STATS_FILE} — "
                "decontamination overlap rate is unlogged for this arm)"
            )
    else:
        if doc_batches is None:
            raise ValueError(f"{arm.value}: no doc-batch factory and no prebuilt shard dir")
        shard_dir = work_dir / arm.value
        _, built = build_corpus_shard(
            arm,
            doc_batches,
            shard_dir,
            vocab_size=vocab_size,
            max_train_bytes=max_train_bytes,
            target_tokens=target_tokens,
            held_out=held_out,
            decontaminate=decontaminate,
            extra_eval_paths=extra_eval_paths,
            num_workers=num_workers,
            progress=progress,
        )
        stats = built
        _write_build_stats(shard_dir, stats)
    tokens = load_dataset_tokens(shard_dir)
    tokenizer = load_tokenizer(shard_dir)

    model_cfg = model_config_for_depth(depth, len(tokenizer.vocab), train_cfg.context_length)
    seed_everything(train_cfg.seed)
    model = TransformerLM(model_cfg)

    held_ids = encode_held_out(tokenizer, held_out)
    init_ce = loss_at_init(
        model, held_ids, context_length=train_cfg.context_length, device=train_cfg.device
    )

    # Curve discipline (optimizer_race pattern): fixed sequential val windows, no RNG — a
    # tail slice of the arm's own stream, disjoint from nothing but held by the eval being
    # in-corpus (its only job is curve stability, not the verdict).
    val_len = min(train_cfg.context_length * (train_cfg.eval_batches + 1), tokens.size // 2)
    val = np.asarray(tokens[-val_len:], dtype=np.int64)
    tokens_per_step = train_cfg.batch_size * train_cfg.context_length
    val_curve: list[tuple[float, float]] = []

    def _hook(step: int, val_ce: float) -> None:
        val_curve.append(((step + 1) * tokens_per_step, val_ce))

    history = train(train_cfg, tokens, model, val_data=val, eval_hook=_hook)

    bpb = score_held_out_bpb(
        model,
        tokenizer,
        held_out,
        context_length=train_cfg.context_length,
        device=train_cfg.device,
    )
    core: CoreSuiteResult | None = None
    if core_specs is not None:
        core = evaluate_core_suite(
            model,
            tokenizer,
            core_specs,
            device=train_cfg.device,
            limit=core_limit,
            bos_id=tokenizer.encode(DOC_SEPARATOR)[0],
            max_seq_len=train_cfg.context_length,
        )

    n_docs_total = stats.n_docs + stats.n_docs_filtered if stats is not None else 0
    return CorpusArmResult(
        arm=arm,
        n_params=sum(p.numel() for p in model.parameters()),
        n_tokens=train_cfg.max_steps * tokens_per_step,
        n_shard_tokens=stats.n_tokens if stats is not None else int(tokens.size),
        vocab_size=len(tokenizer.vocab),
        init_val_ce=init_ce,
        bpb=bpb,
        core=core,
        overlap_rate=(
            stats.n_docs_filtered / n_docs_total if stats is not None and n_docs_total else None
        ),
        train_history=history,
        val_curve=val_curve,
        wall_seconds=time.perf_counter() - t0,
    )


@dataclass
class CorpusAblationResult:
    """The F12 verdict: both arms plus the pre-registered bpb comparison on shared bytes."""

    baseline: CorpusArmResult  # FINEWEB_EDU — the banked corpus
    challenger: CorpusArmResult  # CLIMBMIX — the candidate
    held_out_n_docs: int
    held_out_n_bytes: int
    compute_flops: float

    @property
    def bpb_delta(self) -> float:
        """challenger − baseline bpb at iso-FLOP (negative ⇒ ClimbMix wins per byte)."""
        return self.challenger.bpb.bits_per_byte - self.baseline.bpb.bits_per_byte

    @property
    def verdict(self) -> str:
        """The pre-registered F12 decision (bpb-only; CORE is too noisy single-seed)."""
        return "climbmix_wins" if self.bpb_delta < 0 else "keep_fineweb_edu"

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": arm_result_to_dict(self.baseline),
            "challenger": arm_result_to_dict(self.challenger),
            "held_out_n_docs": self.held_out_n_docs,
            "held_out_n_bytes": self.held_out_n_bytes,
            "compute_flops": self.compute_flops,
            "bpb_delta": self.bpb_delta,
            "verdict": self.verdict,
        }


def run_corpus_ablation(
    corpora: Mapping[CorpusAblationArm, Callable[[], Iterable[Sequence[str]]]],
    held_out: HeldOutSet,
    work_dir: str | Path,
    train_cfg: TrainConfig,
    *,
    depth: int = 6,
    vocab_size: int = 32768,
    max_train_bytes: int = DEFAULT_BPE_TRAIN_BYTES,
    target_tokens: int | None = None,
    decontaminate: bool = True,
    extra_eval_paths: Sequence[str | Path] = (),
    core_specs: Sequence[TaskSpec] | None = None,
    core_limit: int | None = None,
    num_workers: int = 0,
    prebuilt: Mapping[CorpusAblationArm, str | Path] | None = None,
    arm_hook: Callable[[CorpusArmResult], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> CorpusAblationResult:
    """Run both arms — same model, steps, seed, recipe; different corpus + retrained BPE — and
    return the pre-registered F12 verdict.

    ``corpora`` maps each arm to a ZERO-ARG doc-batch factory (the stream is consumed twice:
    once truncated for BPE training, once fully for sharding — see ``build_dataset_streaming``;
    :func:`hub_doc_batches` builds one from downloaded hub parquet). Each arm's shards +
    ``tokenizer.json`` land in ``work_dir/<arm>/``. An arm listed in ``prebuilt`` instead
    adopts an already-staged shard dir (banked shards, or a previous run's output — the
    resume path) and needs no factory. ``held_out`` is the shared raw-byte eval set: it is
    13-gram-decontaminated OUT of both arms' shards and BPE samples at build time, and the
    drop rate is logged per arm (``None`` for a prebuilt arm without build stats).
    ``train_cfg`` (one config, both arms — the iso-FLOP contract) needs ``eval_every > 0``
    so each arm returns a stable val curve. ``core_specs=None`` skips CORE (the
    network-backed 22-task suite never loads implicitly). ``arm_hook(result)`` fires the
    moment each arm finishes — the incremental-persistence seam (a multi-hour run must never
    hold its only copy of a finished arm in memory).
    """
    prebuilt = prebuilt or {}
    missing = (
        {CorpusAblationArm.FINEWEB_EDU, CorpusAblationArm.CLIMBMIX} - set(corpora) - set(prebuilt)
    )
    if missing:
        raise ValueError(f"corpora is missing arm(s): {[a.value for a in missing]}")
    if train_cfg.eval_every <= 0:
        raise ValueError(
            "run_corpus_ablation needs train_cfg.eval_every > 0 — the DoD wants a stable "
            "val curve per arm, not just an endpoint"
        )
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    results: dict[CorpusAblationArm, CorpusArmResult] = {}
    for arm in (CorpusAblationArm.FINEWEB_EDU, CorpusAblationArm.CLIMBMIX):
        result = _run_arm(
            arm,
            corpora.get(arm),
            held_out,
            work_dir,
            train_cfg,
            depth=depth,
            vocab_size=vocab_size,
            max_train_bytes=max_train_bytes,
            target_tokens=target_tokens,
            decontaminate=decontaminate,
            extra_eval_paths=extra_eval_paths,
            core_specs=core_specs,
            core_limit=core_limit,
            num_workers=num_workers,
            prebuilt_dir=prebuilt.get(arm),
            progress=progress,
        )
        results[arm] = result
        if arm_hook is not None:
            arm_hook(result)

    baseline = results[CorpusAblationArm.FINEWEB_EDU]
    challenger = results[CorpusAblationArm.CLIMBMIX]
    if baseline.n_tokens != challenger.n_tokens:
        raise ValueError(
            f"token budgets differ (baseline {baseline.n_tokens} vs challenger "
            f"{challenger.n_tokens}): not an iso-FLOP ablation — the verdict is void"
        )
    return CorpusAblationResult(
        baseline=baseline,
        challenger=challenger,
        held_out_n_docs=len(held_out.docs),
        held_out_n_bytes=held_out.n_bytes,
        compute_flops=compute_from_params_tokens(baseline.n_params, baseline.n_tokens),
    )


__all__ = [
    "CORPUS_SOURCES",
    "CorpusAblationArm",
    "CorpusAblationResult",
    "CorpusArmResult",
    "CorpusSource",
    "HeldOutSet",
    "arm_result_to_dict",
    "build_corpus_shard",
    "encode_held_out",
    "hub_doc_batches",
    "loss_at_init",
    "run_corpus_ablation",
    "score_held_out_bpb",
    "split_held_out",
]
