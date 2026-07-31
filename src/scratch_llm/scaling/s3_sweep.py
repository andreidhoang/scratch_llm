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
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TypedDict

import numpy as np

from scratch_llm.scaling.isoflop import (
    PowerLaw,
    check_exponent_sum,
    compute_from_params_tokens,
    fit_powerlaw,
    isoflop_min,
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

    point: str  # "s1".."s8"
    depth: int
    n_params: int  # exact, measured at instantiation
    ratio: int  # D:N
    tokens: int  # D = ratio × N (the planned budget)
    compute: float  # C = 6·N·D
    val_loss: float  # final val CE per token (nats) on the held-out tail slice
    val_bpb: float  # final val bits-per-byte — the fit's loss axis
    wall_s: float


@dataclass(frozen=True)
class GridPoint:
    """One IsoFLOP grid point. N is exact (instantiated), D = ratio × N, C = 6·N·D."""

    point: str
    depth: int
    ratio: int
    n_params: int
    tokens: int
    compute: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "point": self.point,
            "depth": self.depth,
            "n_params": self.n_params,
            "ratio": self.ratio,
            "tokens": self.tokens,
            "compute": self.compute,
        }


@cache
def _instantiated_params(depth: int, vocab_size: int, context_length: int) -> int:
    """Exact N for a depth: build the model and count — the plan table's approximations are
    never trusted (optimizer_race.py uses the same instantiation-count pattern)."""
    from scratch_llm.model import TransformerLM

    model = TransformerLM(model_config_for_depth(depth, vocab_size, context_length))
    return sum(p.numel() for p in model.parameters())


def build_grid(
    vocab_size: int = VOCAB_SIZE, context_length: int = CONTEXT_LENGTH
) -> list[GridPoint]:
    """Instantiate the s1–s8 grid: N from ``model_config_for_depth``, D = ratio × N, C = 6ND."""
    grid: list[GridPoint] = []
    for point, depth, ratio in GRID_SPEC:
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


def select_points(grid: list[GridPoint], names: str) -> list[GridPoint]:
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


def run_point(
    point: GridPoint,
    *,
    data_dir: str | Path,
    batch_size: int,
    device: str = "cpu",
    seed: int = 0,
    bf16: bool = False,
    compile_model: bool = False,
) -> SweepRecord:
    """Run one grid point on the real-corpus pretrain path (speedrun's shard-backed chain:
    shards → pretrain → eval) and return its record.

    The token budget D = ratio × N becomes a step count via ``steps_for_budget`` (steps =
    D / (batch × ctx)); val_bpb and val_loss (nats/token) come from speedrun's report card on
    the held-out tail slice, computed by ``eval.metrics.bits_per_byte``.
    """
    steps = steps_for_budget(point.tokens, batch_size, CONTEXT_LENGTH)
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
    return SweepRecord(
        point=point.point,
        depth=point.depth,
        n_params=res.n_params,
        ratio=point.ratio,
        tokens=point.tokens,
        compute=point.compute,
        val_loss=card.nats_per_token,
        val_bpb=card.val_bpb,
        wall_s=res.seconds,
    )


def run_sweep(
    points: list[GridPoint],
    *,
    data_dir: str | Path,
    out_dir: str | Path,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 0,
    bf16: bool = False,
    compile_model: bool = False,
) -> list[SweepRecord]:
    """Run the selected grid points in order, appending each finished record to
    ``<out_dir>/results.json`` as it lands (a crash mid-sweep keeps the finished points)."""
    results_path = Path(out_dir) / RESULTS_FILENAME
    records: list[SweepRecord] = []
    for point in points:
        steps = steps_for_budget(point.tokens, batch_size, CONTEXT_LENGTH)
        print(
            f"[{point.point}] depth={point.depth} N={point.n_params:,} ratio={point.ratio} "
            f"D={point.tokens:,} tok C={point.compute:.2e} -> {steps} steps "
            f"(D / (batch {batch_size} x ctx {CONTEXT_LENGTH}))"
        )
        record = run_point(
            point,
            data_dir=data_dir,
            batch_size=batch_size,
            device=device,
            seed=seed,
            bf16=bf16,
            compile_model=compile_model,
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
    n_opts: list[float]  # argmin-bpb N per budget
    d_opts: list[float]  # the argmin run's RECORDED token budget per budget (see module docstring)
    d_opts_bridge: list[float]  # C/(6·N_opt) consistency diagnostic — must track d_opts
    n_law: PowerLaw
    d_law: PowerLaw
    r2_n: float
    r2_d: float
    optimal_ratio_at_d20: float  # D_opt/N_opt extrapolated to the d20's C
    recommend_reregister: bool  # optimal_ratio_at_d20 < DN_DECISION_THRESHOLD
    oracle_core_at_d20: float  # nanochat's CORE fit evaluated at the d20's C

    def to_dict(self) -> dict[str, object]:
        return {
            "n_law": {"exponent": self.n_law.exponent, "coeff": self.n_law.coeff},
            "d_law": {"exponent": self.d_law.exponent, "coeff": self.d_law.coeff},
            "exponent_sum": self.n_law.exponent + self.d_law.exponent,
            "r2_loglog": {"n_law": self.r2_n, "d_law": self.r2_d, "gate": R2_GATE},
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
            lines.append(
                f"| {r['point']} | {r['depth']} | {r['n_params']:,} | {r['ratio']} "
                f"| {r['tokens']:,} | {r['compute']:.2e} | {r['val_bpb']:.4f} "
                f"| {r['val_loss']:.4f} | {r['wall_s'] / 3600:.2f} |"
            )
        lines += [
            "",
            "## Power-law fit (per-budget argmin-bpb, log-log)",
            "",
            f"- N_opt = {self.n_law.coeff:.4f} · C^{a:.4f} (R² = {self.r2_n:.4f})",
            f"- D_opt = {self.d_law.coeff:.4f} · C^{b:.4f} (R² = {self.r2_d:.4f})",
            f"- gate a+b ∈ [0.95, 1.05]: a+b = {a + b:.4f} — PASS",
            f"- gate R² ≥ {R2_GATE}: min R² = {min(self.r2_n, self.r2_d):.4f} — PASS",
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


def fit_scaling_law(records: list[SweepRecord]) -> FitReport:
    """Fit ``N_opt ∝ C^a`` / ``D_opt ∝ C^b`` from finished sweep records, gates first.

    Min-pick is on **val_bpb** (never raw CE — cross-config comparability). D_opt is the
    min-picked run's recorded token budget, so ``check_exponent_sum`` genuinely checks the
    driver's C=6ND wiring instead of passing identically (see the module docstring). Both
    gates raise before any extrapolation.
    """
    if len(records) < 2:
        raise ValueError(f"need >=2 finished grid points to fit, got {len(records)}")
    runs = [
        {
            "compute_budget": float(r["compute"]),
            "final_loss": float(r["val_bpb"]),  # the fit axis is bpb
            "parameters": float(r["n_params"]),
            "tokens": float(r["tokens"]),
        }
        for r in records
    ]
    pairs = isoflop_min(runs)
    budgets = [c for c, _ in pairs]
    n_opts = [n for _, n in pairs]
    # isoflop_min returns only (C, N_opt); recover each argmin run to read its recorded D.
    d_opts = [
        next(r["tokens"] for r in runs if r["compute_budget"] == c and r["parameters"] == n)
        for c, n in pairs
    ]
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
