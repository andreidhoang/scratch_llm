"""S3 scaling-law sweep — the real-corpus IsoFLOP grid driver (E2E §S3).

S3 fits OUR scaling law on OUR data/tokenizer instead of borrowing nanochat's published curve
and Chinchilla's 20:1 folklore: eight grid points (depths {4, 8, 12} × D:N ratios {8, 20, 40}
minus the (12, 40) point) run on the real-corpus pretrain path (``speedrun --depth``), and the
existing fitter (:mod:`scratch_llm.scaling.isoflop`) turns per-budget argmin-bpb runs into
``N_opt ∝ C^a`` / ``D_opt ∝ C^b``.

Three disciplines, all pre-registered in E2E §S3(c)/(e):

* **Exact N from instantiation** — ``model_config_for_depth`` + ``TransformerLM`` parameter
  count, never the plan table's "~20M" approximations (d4 = 19,990,784 at vocab 32768).
* **Fit in bpb, never raw CE** — bpb is tokenizer-invariant and low-noise (CORE's run-to-run
  spread at this scale is ±0.008–0.016, so CORE never enters the fit).
* **Gates fire loudly before any extrapolation** — ``check_exponent_sum`` (a+b ∈ [0.95, 1.05],
  the C=6ND identity) and an R² ≥ 0.98 log-log gate; a failure means "no clean power law at
  our scale", itself the reportable result.

One deliberate wiring choice: ``D_opt`` per budget is the min-picked run's **recorded** token
budget, not the ``C/(6N)`` bridge. Bridge-derived D makes a+b ≡ 1 identically (the regression
lock in tests/test_scaling.py), which would mute exactly the driver bugs the gate exists to
catch — a wrong steps conversion or a misaligned record shifts the recorded D off the bridge
and the sum screams. The bridge (``tokens_from_compute_params``) is still computed and reported
as a consistency diagnostic.

The v2 amendment (the pre-registered repair after the s1–s7 fit failed its R² gate at 0.7709):
the (depth × ratio) grid gives ONE model size per compute budget, so the per-budget min-pick
degenerates to "the only run wins" and no power law can be checked. ``GRID_SPEC_V2`` swaps to a
**budget × size** grid — five exact-C budgets × three depths each (b1_d4 .. b5_d16, depths
6/14/16 included; ``model_config_for_depth`` is generic over depth, d_model = 64·depth) — with
three disciplines of its own:

* **Exact-C coincidence** — per point, D = C_target/(6·N) with the exact instantiated N, rounded
  to a whole number of optimizer steps (nearest step, so effective C coincides with the target
  up to ≤ half a step's worth of FLOPs), then the effective C is RECOMPUTED from the rounded D
  so every record stays exactly self-consistent (``C_eff = 6·N·D``). Both target and effective
  C are recorded.
* **Quadratic min-pick** — ``isoflop_min_smooth`` (Chinchilla Approach 2) fits a quadratic in
  log N per budget and takes the interpolated vertex; ``D_opt`` stays a RECORDED token budget
  (the sampled run nearest the vertex in log N), never the bridge, so the a+b gate keeps
  checking driver wiring instead of passing identically.
* **Dual-seed averaging** — records carry their seed; replicate seeds of the same
  (point, budget) are averaged on the bpb axis BEFORE any min-pick.

The v1 path (``GRID_SPEC``, raw ``isoflop_min``, ``--no-smooth``) is kept intact for s1–s8
reproducibility.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import numpy as np

from scratch_llm.scaling.isoflop import (
    PowerLaw,
    check_exponent_sum,
    compute_from_params_tokens,
    fit_powerlaw,
    isoflop_min,
    isoflop_min_smooth,
    tokens_from_compute_params,
)
from scratch_llm.speedrun import SpeedrunConfig, model_config_for_depth, run_speedrun

# The adopted recipe's fixed axes (E2E §S3(c)): the staged tokenizer's vocab and bf16 ctx 2048.
VOCAB_SIZE = 32768
CONTEXT_LENGTH = 2048

# The grid: depths {4, 8, 12} × D:N ratios {8, 20, 40} minus the (12, 40) point — s8 (d12 @
# ratio-20, C ≈ 2.2e18) is the largest point and the first to drop if the budget trips.
GRID_SPEC: tuple[tuple[str, int, int], ...] = (
    ("s1", 4, 8),
    ("s2", 4, 20),
    ("s3", 4, 40),
    ("s4", 8, 8),
    ("s5", 8, 20),
    ("s6", 8, 40),
    ("s7", 12, 8),
    ("s8", 12, 20),
)

# The v3 recipe/scale-extension grid (FRONTIER_2026_D20_CERTAINTY_PLAN §6 P5 protocol +
# FRONTIER_2026_SCALING_PROGRAM stage 2): ratio-4 points for the P5 LR sweep (d12 core sweep,
# d8 width probe) and the d14 rungs that extend the ladder's leverage toward the d20. The P5
# confirmation point (d12 @ ratio-8) and S3.5's d12-r20 are the v1 grid's s7/s8 — reused, not
# duplicated. LR is NOT a grid axis: one point per invocation, ``--lr`` carries the multiplier.
GRID_SPEC_V3: tuple[tuple[str, int, int], ...] = (
    ("p5_d12r4", 12, 4),
    ("p5_d8r4", 8, 4),
    ("s35_d14r8", 14, 8),
    ("s35_d14r20", 14, 20),
)

# The v2 budget × size grid (the R²-gate repair): (target budget C, depths per budget). Each
# budget gets THREE model sizes so the per-budget min-pick can interpolate instead of
# degenerating. Points are named b<budget index>_d<depth> — b1_d4 .. b5_d16. Sizes are staggered
# so the interior budgets share depths (d8/d12/d14 repeat across budgets) and the vertex stays
# bracketed: small budgets center on shallow models, large budgets on deep ones.
GRID_SPEC_V2: tuple[tuple[float, tuple[int, ...]], ...] = (
    (1.0e17, (4, 6, 8)),
    (4.0e17, (6, 8, 12)),
    (8.5e17, (8, 12, 14)),
    (2.2e18, (8, 12, 14)),
    (4.5e18, (12, 14, 16)),
)

# Pre-registered gates (E2E §S3(e)).
R2_GATE = 0.98
DN_DECISION_THRESHOLD = 15.0  # measured compute-optimal D:N below this ⇒ re-register the d20's D

# The d20 the fit extrapolates to: 480.4M params × 9.6B registered tokens ≈ 2.77e19 FLOPs.
D20_COMPUTE_FLOPS = 2.77e19
D20_REGISTERED_TOKENS = 9.6e9

# External oracle overlay (E2E §S3(d)) — nanochat's published points, plotted on the same axes.
ORACLE_POINTS: dict[str, dict[str, float]] = {
    # nanochat d20-miniseries: 477M params / 3.82B tokens / CORE 0.1708 (discussion #420).
    "nanochat_d20_miniseries": {"n_params": 477e6, "tokens": 3.82e9, "core": 0.1708},
    # The leaderboard line: d24 ratio-8 FP8 ClimbMix, CORE 0.2626 / val_bpb 0.718.
    "nanochat_leaderboard_d24_ratio8": {"core": 0.2626, "val_bpb": 0.718},
    # The GPT-2 XL anchor the leaderboard is scored against.
    "gpt2_xl": {"core": 0.256525},
}
# nanochat's published CORE fit: CORE = 1 − 3.7555·FLOPs^(−0.0344).
CORE_FIT_COEFF = 3.7555
CORE_FIT_EXPONENT = -0.0344

RESULTS_FILENAME = "results.json"
FIT_JSON_FILENAME = "fit.json"
FIT_MD_FILENAME = "fit.md"


class SweepRecord(TypedDict):
    """One finished grid point — appended to ``results.json`` by the ``run`` subcommand."""

    point: str  # "s1".."s8" (v1) or "b1_d4".."b5_d16" (v2)
    depth: int
    n_params: int  # exact, measured at instantiation
    ratio: float  # planned D:N (v1) / effective D:N = tokens/n_params (v2)
    tokens: int  # D — v1: ratio × N; v2: C_target/(6N) rounded to whole optimizer steps
    compute: float  # effective C = 6·N·D, recomputed from the recorded D (self-consistent)
    val_loss: float  # final val CE per token (nats) on the held-out tail slice
    val_bpb: float  # final val bits-per-byte — the fit's loss axis
    wall_s: float
    seed: int  # the run's seed; replicates of one (point, budget) average before min-pick
    target_compute: NotRequired[float]  # v2 only: the registered budget C_k this point serves
    lr: NotRequired[float]  # peak LR override (--lr); absent = the recipe default 3e-3


@dataclass(frozen=True)
class GridPoint:
    """One IsoFLOP grid point. N is exact (instantiated), D = ratio × N, C = 6·N·D."""

    point: str
    depth: int
    ratio: int
    n_params: int
    tokens: int
    compute: float

    def planned_steps(self, batch_size: int, context_length: int) -> int:
        """steps = D / (batch × ctx), ceil — the run never undershoots the token budget."""
        return steps_for_budget(self.tokens, batch_size, context_length)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "point": self.point,
            "depth": self.depth,
            "n_params": self.n_params,
            "ratio": self.ratio,
            "tokens": self.tokens,
            "compute": self.compute,
        }


@dataclass(frozen=True)
class BudgetGridPoint:
    """One v2 exact-C grid point: N exact (instantiated); D = C_target/(6·N) rounded to a whole
    number of optimizer steps; ``compute`` is the EFFECTIVE C recomputed from the rounded D, so
    the record stays exactly self-consistent while budgets coincide across sizes up to the
    per-size step rounding."""

    point: str  # "b<budget index>_d<depth>"
    depth: int
    n_params: int
    target_compute: float  # the registered budget C_k
    tokens: int  # D = steps × batch × ctx
    compute: float  # effective C = 6·N·D (within half a step's FLOPs of target_compute)
    steps: int  # the whole-step count D was rounded to
    batch_size: int  # the batch the step rounding assumed — running with another batch re-plans

    @property
    def ratio(self) -> float:
        """Effective D:N = tokens/n_params — not planned, a consequence of the exact-C budget."""
        return self.tokens / self.n_params

    def planned_steps(self, batch_size: int, context_length: int) -> int:
        """The steps frozen at plan time; a different batch would change the effective C, so it
        raises rather than silently breaking the record's self-consistency."""
        if batch_size != self.batch_size:
            raise ValueError(
                f"{self.point}: planned for batch {self.batch_size}, got {batch_size} — "
                f"re-plan with build_grid_v2(batch_size={batch_size})"
            )
        return self.steps

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "point": self.point,
            "depth": self.depth,
            "n_params": self.n_params,
            "target_compute": self.target_compute,
            "tokens": self.tokens,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "compute": self.compute,
        }


@cache
def _instantiated_params(depth: int, vocab_size: int, context_length: int) -> int:
    """Exact N for a depth: build the model and count — the plan table's approximations are
    never trusted (optimizer_race.py uses the same instantiation-count pattern).

    The real GPU training path turns on ``qk_norm`` (F9 guardrail) and ``use_sdpa`` (OOM guard),
    so the planned param count must include ``qk_norm``'s extra parameters.
    """
    from dataclasses import replace

    from scratch_llm.model import TransformerLM

    cfg = model_config_for_depth(depth, vocab_size, context_length)
    cfg = replace(cfg, qk_norm=True, use_sdpa=True)
    model = TransformerLM(cfg)
    return sum(p.numel() for p in model.parameters())


def build_grid(
    vocab_size: int = VOCAB_SIZE,
    context_length: int = CONTEXT_LENGTH,
    spec: tuple[tuple[str, int, int], ...] = GRID_SPEC,
) -> list[GridPoint]:
    """Instantiate a depth × ratio grid: N from ``model_config_for_depth``, D = ratio × N,
    C = 6ND. Defaults to the s1–s8 v1 spec; the v3 P5/S3.5 extension passes its own."""
    grid: list[GridPoint] = []
    for point, depth, ratio in spec:
        n_params = _instantiated_params(depth, vocab_size, context_length)
        tokens = ratio * n_params
        grid.append(
            GridPoint(
                point=point,
                depth=depth,
                ratio=ratio,
                n_params=n_params,
                tokens=tokens,
                compute=compute_from_params_tokens(n_params, tokens),
            )
        )
    return grid


def build_grid_v3(
    vocab_size: int = VOCAB_SIZE, context_length: int = CONTEXT_LENGTH
) -> list[GridPoint]:
    """Instantiate the v3 P5/S3.5 extension grid — same N/D/C construction as v1."""
    return build_grid(vocab_size, context_length, spec=GRID_SPEC_V3)


def build_grid_v2(
    batch_size: int = 32, vocab_size: int = VOCAB_SIZE, context_length: int = CONTEXT_LENGTH
) -> list[BudgetGridPoint]:
    """Instantiate the v2 budget × size grid: per point, D = C_target/(6·N) with the exact
    instantiated N, rounded to the NEAREST whole optimizer step (batch × ctx tokens each —
    nearest, not the v1 path's ceil: coincidence with the target budget beats the
    never-undershoot bias here, and the effective C is recomputed from the rounded D so nothing
    is misreported). Depths 6/14/16 need no special-casing — ``model_config_for_depth`` derives
    width/heads from any depth (d_model = 64·depth, head_dim 128)."""
    grid: list[BudgetGridPoint] = []
    for budget_idx, (target_compute, depths) in enumerate(GRID_SPEC_V2, start=1):
        for depth in depths:
            n_params = _instantiated_params(depth, vocab_size, context_length)
            ideal_tokens = tokens_from_compute_params(target_compute, n_params)
            steps = max(1, round(ideal_tokens / (batch_size * context_length)))
            tokens = steps * batch_size * context_length
            grid.append(
                BudgetGridPoint(
                    point=f"b{budget_idx}_d{depth}",
                    depth=depth,
                    n_params=n_params,
                    target_compute=target_compute,
                    tokens=tokens,
                    compute=compute_from_params_tokens(n_params, tokens),
                    steps=steps,
                    batch_size=batch_size,
                )
            )
    return grid


AnyGridPoint = GridPoint | BudgetGridPoint


def select_points(grid: Sequence[AnyGridPoint], names: str) -> list[AnyGridPoint]:
    """Filter the grid to a comma-separated ``--points s1,s2,...`` selection (unknown names
    raise — a typo must not silently shrink the sweep)."""
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    known = {g.point for g in grid}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(f"unknown grid points {unknown} (known: {sorted(known)})")
    return [g for g in grid if g.point in wanted]


def steps_for_budget(tokens: int, batch_size: int, context_length: int) -> int:
    """steps = D / (batch × ctx) — each optimizer step consumes batch_size × context_length
    tokens, so covering the token budget D = ratio × N takes D/(batch×ctx) steps. Ceil so the
    run never undershoots the budget (overshoot is < one step = batch×ctx tokens, ≪1% of D at
    sweep scale)."""
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    return math.ceil(tokens / (batch_size * context_length))


def load_results(path: str | Path) -> list[SweepRecord]:
    """Load ``results.json`` (empty list if absent — the first ``run`` creates it)."""
    p = Path(path)
    if not p.exists():
        return []
    return list(json.loads(p.read_text()))


def append_result(path: str | Path, record: SweepRecord) -> None:
    """Append one finished point's record to ``results.json`` (created on first use)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = load_results(p)
    records.append(record)
    p.write_text(json.dumps(records, indent=2) + "\n")


def average_seed_replicates(records: list[SweepRecord]) -> list[SweepRecord]:
    """Collapse replicate seeds of the same (point, budget) into one record whose val_bpb /
    val_loss / wall_s are the seed MEAN — the averaging happens on the loss axis BEFORE any
    min-pick, so a lucky seed can't win its budget on noise. Records without a ``seed`` field
    (pre-amendment s1–s8 results.json) are treated as seed 0; the budget key is the v2 record's
    ``target_compute`` when present, else the effective ``compute``. First-seen order and the
    first replicate's scaffold (N, D, C — identical across seeds by construction) are kept.
    """
    groups: dict[tuple[str, float], list[SweepRecord]] = {}
    for r in records:
        budget = float(r.get("target_compute", r["compute"]))
        groups.setdefault((r["point"], budget), []).append(r)
    averaged: list[SweepRecord] = []
    for group in groups.values():
        if len(group) == 1:
            averaged.append(group[0])
            continue
        merged = dict(group[0])
        for key in ("val_bpb", "val_loss", "wall_s"):
            merged[key] = float(np.mean([r[key] for r in group]))
        averaged.append(cast(SweepRecord, merged))
    return averaged


def run_point(
    point: GridPoint | BudgetGridPoint,
    *,
    data_dir: str | Path,
    batch_size: int,
    device: str = "cpu",
    seed: int = 0,
    bf16: bool = False,
    compile_model: bool = False,
    lr: float | None = None,
) -> SweepRecord:
    """Run one grid point on the real-corpus pretrain path (speedrun's shard-backed chain:
    shards → pretrain → eval) and return its record.

    The point's token budget becomes a step count via ``planned_steps`` (v1: ceil of
    D/(batch×ctx); v2: the plan-frozen whole steps, batch-mismatches raise); val_bpb and
    val_loss (nats/token) come from speedrun's report card on the held-out tail slice, computed
    by ``eval.metrics.bits_per_byte``. The record carries the seed (dual-seed fits average
    replicates before min-picking) and, for v2 points, the target budget C_k. ``lr`` overrides
    the peak LR (the P5 recipe sweep): the schedule SHAPE (warmup, decay profile) is unchanged,
    only the peak moves, and the override lands in the record so the sweep stays auditable.
    """
    steps = point.planned_steps(batch_size, CONTEXT_LENGTH)
    cfg = SpeedrunConfig(
        depth=point.depth,
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        train_steps=steps,
        batch_size=batch_size,
        data_dir=str(data_dir),
        device=device,
        seed=seed,
        amp_dtype="bf16" if bf16 else None,
        compile=compile_model,
        # SpeedrunConfig.lr (3e-3) is the recipe default, read off the dataclass so this
        # never duplicates it; only P5's --lr overrides the peak.
        lr=lr if lr is not None else SpeedrunConfig.lr,
    )
    res = run_speedrun(cfg)
    if res.n_params != point.n_params:
        raise RuntimeError(
            f"{point.point}: instantiated {res.n_params} params but the grid planned "
            f"{point.n_params} — model_config_for_depth drifted from the plan"
        )
    card = res.report_card
    if card.val_bpb is None or card.nats_per_token is None:
        raise RuntimeError(f"{point.point}: report card carried no val_bpb — the eval stage ran?")
    record = SweepRecord(
        point=point.point,
        depth=point.depth,
        n_params=res.n_params,
        ratio=point.ratio,
        tokens=point.tokens,
        compute=point.compute,
        val_loss=card.nats_per_token,
        val_bpb=card.val_bpb,
        wall_s=res.seconds,
        seed=seed,
    )
    if isinstance(point, BudgetGridPoint):
        record["target_compute"] = point.target_compute
    if lr is not None:
        record["lr"] = lr
    return record


def run_sweep(
    points: Sequence[AnyGridPoint],
    *,
    data_dir: str | Path,
    out_dir: str | Path,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 0,
    bf16: bool = False,
    compile_model: bool = False,
    lr: float | None = None,
) -> list[SweepRecord]:
    """Run the selected grid points in order, appending each finished record to
    ``<out_dir>/results.json`` as it lands (a crash mid-sweep keeps the finished points).
    Dual-seed coverage is one invocation per seed (``--seed 0``, then ``--seed 1``); records
    carry their seed and the fitter averages replicates before min-picking. ``lr`` overrides
    the peak LR for every point in the invocation (P5: one LR per invocation, one out-dir per
    LR, so same-point records at different peaks never share a results.json)."""
    results_path = Path(out_dir) / RESULTS_FILENAME
    records: list[SweepRecord] = []
    for point in points:
        steps = point.planned_steps(batch_size, CONTEXT_LENGTH)
        if isinstance(point, BudgetGridPoint):
            c_desc = f"C_target={point.target_compute:.2e} C_eff={point.compute:.2e}"
        else:
            c_desc = f"C={point.compute:.2e}"
        lr_desc = f" lr={lr:g}" if lr is not None else ""
        print(
            f"[{point.point}] depth={point.depth} N={point.n_params:,} "
            f"D={point.tokens:,} tok (D:N={point.ratio:.2f}) {c_desc} -> {steps} steps "
            f"(batch {batch_size} x ctx {CONTEXT_LENGTH}, seed {seed}{lr_desc})"
        )
        record = run_point(
            point,
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            seed=seed,
            bf16=bf16,
            compile_model=compile_model,
            lr=lr,
        )
        append_result(results_path, record)
        records.append(record)
        print(
            f"[{point.point}] val_bpb={record['val_bpb']:.4f} "
            f"val_loss={record['val_loss']:.4f} wall={record['wall_s'] / 3600:.2f} h "
            f"-> {results_path}"
        )
    return records


# -----------------------------------------------------------------------------------------------
# fit — per-budget min-pick, the two power laws, the gates, the D:N decision rule, the oracle.
# -----------------------------------------------------------------------------------------------


def r2_loglog(law: PowerLaw, xs: list[float], ys: list[float]) -> float:
    """R² of ``law`` against ``(xs, ys)`` in log-log space — the space the fit lives in (a
    raw-space R² would be dominated by the largest budget and bless a bad exponent)."""
    log_x = np.log(np.asarray(xs, dtype=np.float64))
    log_y = np.log(np.asarray(ys, dtype=np.float64))
    pred = np.log(law.coeff) + law.exponent * log_x
    ss_res = float(np.square(log_y - pred).sum())
    ss_tot = float(np.square(log_y - log_y.mean()).sum())
    return 1.0 - ss_res / ss_tot


def check_r_squared(r2: float, gate: float = R2_GATE) -> None:
    """The R² gate — a fit explaining <98% of log-log variance is not a power law we may
    extrapolate; "no clean power law at our scale" is itself the reportable result (§S3(e))."""
    if r2 < gate:
        raise ValueError(
            f"R² gate failed: log-log R²={r2:.4f} < {gate} — no clean power law at our scale; "
            f"report the non-fit instead of extrapolating (trigger T1: add the d14 rung)"
        )


def nanochat_core_fit(compute: float) -> float:
    """nanochat's published CORE fit ``1 − 3.7555·FLOPs^(−0.0344)`` — ≈0.195 at our d20's C."""
    return 1.0 - CORE_FIT_COEFF * compute**CORE_FIT_EXPONENT


@dataclass(frozen=True)
class FitReport:
    """The S3 fit: both laws, both gates' numbers, the D:N decision, and the oracle overlay."""

    budgets: list[float]  # the distinct per-budget C values that survived min-picking
    n_opts: list[float]  # min-picked N per budget (interpolated vertex when smooth)
    d_opts: list[float]  # RECORDED token budget per budget — never the bridge (module docstring)
    d_opts_bridge: list[float]  # C/(6·N_opt) consistency diagnostic — must track d_opts
    n_law: PowerLaw
    d_law: PowerLaw
    r2_n: float
    r2_d: float
    optimal_ratio_at_d20: float  # D_opt/N_opt extrapolated to the d20's C
    recommend_reregister: bool  # optimal_ratio_at_d20 < DN_DECISION_THRESHOLD
    oracle_core_at_d20: float  # nanochat's CORE fit evaluated at the d20's C
    smooth: bool  # True ⇒ quadratic-in-log-N min-pick (Chinchilla A2); False ⇒ raw argmin
    clamped_budgets: list[float]  # budgets whose vertex fell outside the sampled log-N range

    def to_dict(self) -> dict[str, object]:
        return {
            "n_law": {"exponent": self.n_law.exponent, "coeff": self.n_law.coeff},
            "d_law": {"exponent": self.d_law.exponent, "coeff": self.d_law.coeff},
            "exponent_sum": self.n_law.exponent + self.d_law.exponent,
            "r2_loglog": {"n_law": self.r2_n, "d_law": self.r2_d, "gate": R2_GATE},
            "min_pick": {"smooth": self.smooth, "clamped_budgets": self.clamped_budgets},
            "budgets": self.budgets,
            "n_opts": self.n_opts,
            "d_opts": self.d_opts,
            "d_opts_bridge": self.d_opts_bridge,
            "decision_rule": {
                "threshold": DN_DECISION_THRESHOLD,
                "d20_compute_flops": D20_COMPUTE_FLOPS,
                "d20_registered_tokens": D20_REGISTERED_TOKENS,
                "optimal_ratio_at_d20": self.optimal_ratio_at_d20,
                "recommend_reregister": self.recommend_reregister,
            },
            "oracle": {
                **ORACLE_POINTS,
                "nanochat_core_fit": {"coeff": CORE_FIT_COEFF, "exponent": CORE_FIT_EXPONENT},
                "nanochat_core_fit_at_d20": self.oracle_core_at_d20,
            },
        }

    def to_markdown(self, records: list[SweepRecord]) -> str:
        a, b = self.n_law.exponent, self.d_law.exponent
        lines = [
            "# S3 scaling-law sweep — fit summary",
            "",
            "## Grid results (fit axis: val_bpb)",
            "",
            "| point | depth | N | D:N | D (tok) | C (FLOPs) | val_bpb | val_loss | wall (h) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in records:
            ratio = r["ratio"]
            ratio_s = str(int(ratio)) if float(ratio).is_integer() else f"{ratio:.2f}"
            lines.append(
                f"| {r['point']} | {r['depth']} | {r['n_params']:,} | {ratio_s} "
                f"| {r['tokens']:,} | {r['compute']:.2e} | {r['val_bpb']:.4f} "
                f"| {r['val_loss']:.4f} | {r['wall_s'] / 3600:.2f} |"
            )
        pick_mode = "quadratic-in-log-N vertex (Chinchilla A2)" if self.smooth else "raw argmin"
        lines += [
            "",
            f"## Power-law fit (per-budget {pick_mode} on bpb, log-log)",
            "",
            f"- N_opt = {self.n_law.coeff:.4f} · C^{a:.4f} (R² = {self.r2_n:.4f})",
            f"- D_opt = {self.d_law.coeff:.4f} · C^{b:.4f} (R² = {self.r2_d:.4f})",
            f"- gate a+b ∈ [0.95, 1.05]: a+b = {a + b:.4f} — PASS",
            f"- gate R² ≥ {R2_GATE}: min R² = {min(self.r2_n, self.r2_d):.4f} — PASS",
        ]
        if self.clamped_budgets:
            lines.append(
                f"- WARNING: vertex outside the sampled log-N range at budgets "
                f"{[f'{c:.2e}' for c in self.clamped_budgets]} — clamped to the nearer "
                "endpoint's argmin (an extrapolated vertex is not a measurement; widen those "
                "budgets' size spread)"
            )
        lines += [
            "",
            "## D:N decision rule (threshold 15)",
            "",
            f"At the d20's C = {D20_COMPUTE_FLOPS:.2e} FLOPs the fit gives "
            f"N_opt = {self.n_law.predict(D20_COMPUTE_FLOPS):.2e}, "
            f"D_opt = {self.d_law.predict(D20_COMPUTE_FLOPS):.2e} — a compute-optimal D:N of "
            f"**{self.optimal_ratio_at_d20:.1f}** (registered: {D20_REGISTERED_TOKENS:.1e} tokens "
            f"on 480.4M params ≈ ratio 20).",
            "",
        ]
        if self.recommend_reregister:
            lines.append(
                f"**RE-REGISTER:** measured compute-optimal ratio {self.optimal_ratio_at_d20:.1f} "
                f"< {DN_DECISION_THRESHOLD:.0f} ⇒ move the d20's D away from "
                f"{D20_REGISTERED_TOKENS / 1e9:.1f}B tokens before P5, with the inference-aware "
                "overtraining argument written down (Sardana 2401.00448) — decide, don't inherit."
            )
        else:
            lines.append(
                f"**HOLD:** measured compute-optimal ratio {self.optimal_ratio_at_d20:.1f} ≥ "
                f"{DN_DECISION_THRESHOLD:.0f} ⇒ the registered "
                f"{D20_REGISTERED_TOKENS / 1e9:.1f}B-token budget stands."
            )
        lines += [
            "",
            "## Oracle overlay (nanochat's published points)",
            "",
            f"- d20-miniseries: 477M params / 3.82B tok / CORE "
            f"{ORACLE_POINTS['nanochat_d20_miniseries']['core']}",
            f"- leaderboard d24 ratio-8: CORE {ORACLE_POINTS['nanochat_leaderboard_d24_ratio8']['core']}"
            f" / val_bpb {ORACLE_POINTS['nanochat_leaderboard_d24_ratio8']['val_bpb']}",
            f"- GPT-2 XL anchor: CORE {ORACLE_POINTS['gpt2_xl']['core']}",
            f"- nanochat CORE-fit 1 − {CORE_FIT_COEFF}·FLOPs^({CORE_FIT_EXPONENT}) at our d20's C "
            f"⇒ CORE ≈ **{self.oracle_core_at_d20:.3f}** (pre-registered band 0.19–0.22)",
            "",
        ]
        return "\n".join(lines)


def fit_scaling_law(records: list[SweepRecord], *, smooth: bool = True) -> FitReport:
    """Fit ``N_opt ∝ C^a`` / ``D_opt ∝ C^b`` from finished sweep records, gates first.

    Replicate seeds of one (point, budget) are averaged on the bpb axis first
    (``average_seed_replicates``). Min-pick is on **val_bpb** (never raw CE — cross-config
    comparability): ``smooth=True`` (default) fits a quadratic in log N per budget and takes the
    interpolated vertex (Chinchilla Approach 2 — the v2 grid gives ≥3 sizes per budget so the
    vertex is bracketed); ``smooth=False`` keeps the v1 raw argmin. The budget axis is the v2
    record's ``target_compute`` when present (sizes of one budget coincide by construction),
    else the effective ``compute``. D_opt is a RECORDED token budget per budget — the run
    nearest the pick in log N — so ``check_exponent_sum`` genuinely checks the driver's C=6ND
    wiring instead of passing identically (see the module docstring). Both gates raise before
    any extrapolation.
    """
    records = average_seed_replicates(records)
    if len(records) < 2:
        raise ValueError(f"need >=2 finished grid points to fit, got {len(records)}")
    runs = [
        {
            "compute_budget": float(r.get("target_compute", r["compute"])),
            "final_loss": float(r["val_bpb"]),  # the fit axis is bpb
            "parameters": float(r["n_params"]),
            "tokens": float(r["tokens"]),
        }
        for r in records
    ]

    def recorded_tokens(budget: float, n: float) -> float:
        """The recorded D of the budget's run nearest N in log space — the a+b gate's live
        input: a steps-conversion or record-misalignment bug shifts recorded D off the bridge."""
        return min(
            (r for r in runs if r["compute_budget"] == budget),
            key=lambda r: abs(math.log(r["parameters"]) - math.log(n)),
        )["tokens"]

    clamped_budgets: list[float] = []
    if smooth:
        picks = isoflop_min_smooth(runs)
        budgets = [p.budget for p in picks]
        n_opts = [p.n_opt for p in picks]
        clamped_budgets = [p.budget for p in picks if p.clamped]
        d_opts = [recorded_tokens(p.budget, p.n_opt) for p in picks]
        pairs = list(zip(budgets, n_opts, strict=True))
    else:
        pairs = isoflop_min(runs)
        budgets = [c for c, _ in pairs]
        n_opts = [n for _, n in pairs]
        # isoflop_min returns only (C, N_opt); recover each argmin run to read its recorded D.
        d_opts = [recorded_tokens(c, n) for c, n in pairs]
    d_opts_bridge = [tokens_from_compute_params(c, n) for c, n in pairs]

    n_law = fit_powerlaw(budgets, n_opts)
    d_law = fit_powerlaw(budgets, d_opts)
    a, b = n_law.exponent, d_law.exponent
    check_exponent_sum(a, b)  # a+b ∈ [0.95, 1.05] — raises before any extrapolation
    r2_n = r2_loglog(n_law, budgets, n_opts)
    r2_d = r2_loglog(d_law, budgets, d_opts)
    check_r_squared(min(r2_n, r2_d))

    optimal_ratio = d_law.predict(D20_COMPUTE_FLOPS) / n_law.predict(D20_COMPUTE_FLOPS)
    return FitReport(
        budgets=budgets,
        n_opts=n_opts,
        d_opts=d_opts,
        d_opts_bridge=d_opts_bridge,
        n_law=n_law,
        d_law=d_law,
        r2_n=r2_n,
        r2_d=r2_d,
        optimal_ratio_at_d20=optimal_ratio,
        recommend_reregister=optimal_ratio < DN_DECISION_THRESHOLD,
        oracle_core_at_d20=nanochat_core_fit(D20_COMPUTE_FLOPS),
        smooth=smooth,
        clamped_budgets=clamped_budgets,
    )


def write_fit_outputs(
    report: FitReport, records: list[SweepRecord], out_dir: str | Path
) -> tuple[Path, Path]:
    """Emit ``fit.json`` (machine-readable) + ``fit.md`` (the summary) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / FIT_JSON_FILENAME
    md_path = out / FIT_MD_FILENAME
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    md_path.write_text(report.to_markdown(records))
    return json_path, md_path
