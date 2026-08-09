"""P5.5 — the d20 target-scale LR probe gate (ADR-0020 item 2; certainty plan §6).

The one-shot d20 commits ~$100 at a learning rate that was swept at d12 (η* = 0.0021, batch
16,384 tok/step) and transferred by nanochat's composite rule — whose √B term its own author
labels *"not studied carefully, assumption!"*. The d20's global batch (524,288 tok/step) is 32×
the grid's, so the rule predicts::

    lr_center = η* · √(524288 / 16384) = 0.0021 · √32 ≈ 0.0119

The probe MEASURES that prediction at the exact d20 config before the full budget commits:
three arms at lr_center × {0.7, 1.0, 1.4}, each a complete short run (~5% of the d20's tokens,
cosine schedule over the probe horizon — the same protocol as the P5 sweep's ratio-4 horizon),
ranked by val_bpb on the same held-out tail slice (same data dir + seed ⇒ identical val slice
across arms). This is Karpathy's 320-sweep lesson — *"small-scale tuning doesn't transfer;
validate at target scale"* — applied to a one-shot run, and it doubles as the first measurement
of the √B assumption at 32× batch (a ledger result either way).

Disciplines, mirroring the S3.5 runner:

* **Idempotent arms** — one out-dir per arm; a finished ``results.json`` means skip (a
  preempted arm is re-run, a finished arm is never paid for twice).
* **Pre-registered tie-break** — arms within ``TIE_BAND_BPB`` of the best bpb tie, and the tie
  breaks to the LOWER LR (P5's rule, reused verbatim).
* **Distributed-correct** — ``run_arm`` initializes the process group from torchrun's env
  (``utils/dist_launch.py``, runbook §0.5 G2); per-rank data sharding and ZeRO-2 activate
  inside ``train()`` as they do for the d20 itself. Steps are computed against the GLOBAL
  batch (world_size × batch × ctx) so an arm consumes the same tokens at any world size.
* **No stage checkpoints** — an arm is ~10 min; preemption re-runs the arm (idempotent skip
  only fires on a COMPLETED arm). The full d20's checkpoint-every safety net is unchanged.

Logic lives here (importable by tests); ``scripts/d20_probe.py`` is the thin CLI front-end,
same pattern as ``scripts/s3_scaling_sweep.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from scratch_llm.speedrun import SpeedrunConfig, run_speedrun
from scratch_llm.utils.dist_launch import cleanup, maybe_init_from_torchrun

# --- the registered constants (ADR-0020 + the d20 scorecard in bench/RESULTS.md) -------------
ETA_STAR = 0.0021  # P5 d12 sweep winner [MEASURED, artifacts/p5]
GRID_BATCH_TOKENS = 16_384  # the S3.5 grid's step: batch 8 × ctx 2048 (η*'s native batch)
D20_GLOBAL_BATCH_TOKENS = 524_288  # 8 GPU × 32 seq × 2048 ctx (the d20 scorecard [FACT])
D20_DEPTH = 20
D20_CONTEXT_LENGTH = 2048
D20_PER_GPU_BATCH = 32  # per-rank sequences; global batch = world × this × ctx
D20_VOCAB_SIZE = 32768
D20_REGISTERED_TOKENS = 9_600_000_000  # ratio-20 deliberate overtrain (HOLD is permanent)
PROBE_FRACTION = 0.05  # each arm ≈ 5% of the d20's token budget (ADR-0020: "~0.5B tokens")
PROBE_TOKENS = int(D20_REGISTERED_TOKENS * PROBE_FRACTION)  # 480,000,000
LR_MULTIPLIERS = (0.7, 1.0, 1.4)  # the P5 sweep's bracket, re-centered on the composite rule
TIE_BAND_BPB = 0.003  # P5's pre-registered tie-break band, reused verbatim

RESULTS_FILENAME = "results.json"
SELECTION_FILENAME = "selection.json"


def center_lr() -> float:
    """The composite rule's √B transfer of η* to the d20's global batch: η*·√32 ≈ 0.0119."""
    return ETA_STAR * math.sqrt(D20_GLOBAL_BATCH_TOKENS / GRID_BATCH_TOKENS)


@dataclass(frozen=True)
class ProbeArm:
    """One probe arm: the d20 config at ``center_lr() × mult``."""

    name: str  # "lr0.7" / "lr1.0" / "lr1.4" — also the arm's out-dir name
    mult: float
    lr: float


def build_arms() -> list[ProbeArm]:
    """The three registered arms, in sweep order (cheap insurance first: lowest LR first)."""
    base = center_lr()
    return [ProbeArm(name=f"lr{m}", mult=m, lr=base * m) for m in LR_MULTIPLIERS]


class ProbeRecord(TypedDict):
    """One finished arm — written to ``<out-root>/<arm>/results.json`` by rank 0."""

    arm: str
    mult: float
    lr: float
    depth: int
    n_params: int  # exact, measured at instantiation — must equal the d20's 480,431,360
    tokens: int  # the arm's token budget (steps × global batch, ≥ PROBE_TOKENS by <1 step)
    steps: int
    val_bpb: float  # the selection axis
    val_loss: float  # nats/token, for the ledger
    wall_s: float
    seed: int
    world_size: int  # the batch the steps were planned against — audits the √B measurement


@dataclass(frozen=True)
class ProbeSelection:
    """The gate's verdict: which arm the full-budget run commits at, and who tied."""

    winner: ProbeArm
    winner_bpb: float
    tied: list[str]  # arms within TIE_BAND_BPB of the best (excl. winner); empty = clean win
    records: list[ProbeRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "winner": self.winner.name,
            "winner_lr": self.winner.lr,
            "winner_bpb": self.winner_bpb,
            "tied": self.tied,
            "tie_band_bpb": TIE_BAND_BPB,
            "records": [dict(r) for r in self.records],
        }


def arm_dir(out_root: str | Path, arm: ProbeArm) -> Path:
    return Path(out_root) / arm.name


def load_arm_record(out_root: str | Path, arm: ProbeArm) -> ProbeRecord | None:
    """The arm's finished record, or None if it never completed (absent/empty results.json)."""
    path = arm_dir(out_root, arm) / RESULTS_FILENAME
    if not path.exists() or path.stat().st_size == 0:
        return None
    records = json.loads(path.read_text())
    return ProbeRecord(**records[0]) if records else None


def steps_for_probe(tokens: int, per_step_tokens: int) -> int:
    """steps = tokens / global-batch — ceil, so the arm never undershoots its token budget."""
    if per_step_tokens <= 0:
        raise ValueError("per_step_tokens must be positive")
    return math.ceil(tokens / per_step_tokens)


def run_arm(
    arm: ProbeArm,
    *,
    out_root: str | Path,
    data_dir: str | None = None,
    corpus_path: str | None = None,
    device: str = "cuda",
    batch_size: int = D20_PER_GPU_BATCH,
    context_length: int = D20_CONTEXT_LENGTH,
    vocab_size: int = D20_VOCAB_SIZE,
    seed: int = 0,
    bf16: bool = True,
    compile_model: bool = False,
    depth: int = D20_DEPTH,
    tokens: int = PROBE_TOKENS,
) -> ProbeRecord:
    """Run one probe arm (idempotent) and return its record.

    Under torchrun, every rank trains (ZeRO-2 activates inside ``train()``); rank 0 alone
    writes ``results.json`` — all ranks evaluate the identical model on the identical val
    slice, so rank 0's bpb IS the arm's bpb. A finished arm short-circuits BEFORE any
    process-group init, so re-running a completed arm costs nothing (not even a GPU).
    ``depth``/``tokens``/``vocab_size`` are the d20 registration by default; tests shrink
    them to run the wiring end-to-end on CPU in seconds (the nano pre-flight philosophy).
    """
    existing = load_arm_record(out_root, arm)
    if existing is not None:
        return existing

    ctx = maybe_init_from_torchrun(device)
    try:
        per_step = ctx.world_size * batch_size * context_length
        steps = steps_for_probe(tokens, per_step)
        cfg = SpeedrunConfig(
            depth=depth,
            vocab_size=vocab_size,
            context_length=context_length,
            train_steps=steps,
            batch_size=batch_size,
            lr=arm.lr,
            optimizer="muon_adamw",  # the d20's adopted recipe — the probe changes LR only
            amp_dtype="bf16" if bf16 else None,
            compile=compile_model,
            device=ctx.device,
            seed=seed,
            data_dir=data_dir,
            corpus_path=corpus_path,
        )
        # work_dir stays None on every rank: the probe's crash-safety is arm-level idempotency,
        # not intra-stage snapshots (see the module docstring).
        res = run_speedrun(cfg)
        record = ProbeRecord(
            arm=arm.name,
            mult=arm.mult,
            lr=arm.lr,
            depth=depth,
            n_params=res.n_params,
            tokens=steps * per_step,
            steps=steps,
            val_bpb=_checked_metric(res.report_card.val_bpb, arm),
            val_loss=_checked_metric(res.report_card.nats_per_token, arm),
            wall_s=res.seconds,
            seed=seed,
            world_size=ctx.world_size,
        )
        if ctx.rank == 0:
            out = arm_dir(out_root, arm)
            out.mkdir(parents=True, exist_ok=True)
            (out / RESULTS_FILENAME).write_text(json.dumps([record], indent=2) + "\n")
        return record
    finally:
        cleanup(ctx)


def _checked_metric(value: float | None, arm: ProbeArm) -> float:
    if value is None:
        raise RuntimeError(f"{arm.name}: report card carried no val_bpb — the eval stage ran?")
    return value


def select_winner(
    arms: list[ProbeArm],
    records: list[ProbeRecord],
    tie_band: float = TIE_BAND_BPB,
) -> ProbeSelection:
    """The gate's decision rule: min val_bpb wins; ties within ``tie_band`` break to lower LR.

    Every registered arm must have a record — selecting from a partial probe would silently
    shrink the bracket (a crashed arm is re-run, never skipped past).
    """
    by_arm = {r["arm"]: r for r in records}
    missing = [a.name for a in arms if a.name not in by_arm]
    if missing:
        raise ValueError(f"probe arms missing results: {missing} — re-run them before selecting")
    ordered = [by_arm[a.name] for a in arms]
    best = min(r["val_bpb"] for r in ordered)
    candidates = [r for r in ordered if r["val_bpb"] <= best + tie_band]
    winner_record = min(candidates, key=lambda r: r["lr"])
    winner = next(a for a in arms if a.name == winner_record["arm"])
    tied = [r["arm"] for r in candidates if r["arm"] != winner.name]
    return ProbeSelection(
        winner=winner, winner_bpb=winner_record["val_bpb"], tied=tied, records=ordered
    )


def load_selection(out_root: str | Path) -> ProbeSelection:
    """Select from the arms' on-disk records (the CLI's ``select`` path)."""
    arms = build_arms()
    records = [r for a in arms if (r := load_arm_record(out_root, a)) is not None]
    return select_winner(arms, records)


def write_selection(selection: ProbeSelection, out_root: str | Path) -> Path:
    path = Path(out_root) / SELECTION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection.to_dict(), indent=2) + "\n")
    return path
