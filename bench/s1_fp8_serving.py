"""S1/S-R2 — the measurement entry point: FP8 weights + E4M3 per-head KV against the bf16 engine.

    python bench/s1_fp8_serving.py --floor bf16 --engine <module:factory>   the floor, alone
    python bench/s1_fp8_serving.py --rung  S-R2 --engine <module:factory>   the rung's number
    python bench/s1_fp8_serving.py --rung  S-R2 --dry-run                   print the plan, run nothing

The last line of stdout is always a single bare number — the floor's tokens/s, or the rung's
speedup — and ``experiments/S1/S-R2/{floor,run}.sh`` read exactly that line. Recorded measurements
go through ``infra/bench.sh``, which locks clocks, records provenance, and refuses to run without a
prediction already in the ledger.

The one rule that makes this rung a rung
----------------------------------------
**The speedup is printed only if the quality gate passes.** FP8 is unconditionally faster — one
byte per weight instead of two, one byte per cached K/V element instead of two — so a speedup on
its own is a restatement of the format, not a result. If the gate FAILs, or if it is UNJUDGED
because the thresholds are not yet written (``quant/s_r2_gate.gate_thresholds`` — Huy's hole), this
program prints the evidence and exits non-zero, and no number reaches the ledger.

Both arms are measured in the same process, back to back, under one clock lock, on the same model
and the same shape. That is what "the floor is the bf16 engine at matched model and shape" means
operationally: not a bf16 number from yesterday's run, and not vLLM's number — the same engine with
the quantization switched off.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from scratch_llm.quant.fp8_weights import DEFAULT_FAST_ACCUM, rowwise_scaled_mm_supported
from scratch_llm.quant.s_r2_gate import QualityEvidence, gate_thresholds, verdict

_BENCH_DIR = Path(__file__).resolve().parent

#: What to do on a laptop. Printed by every path that needs silicon and has none.
BOX = (
    "infra/rent.sh sync-up <user@host> && infra/rent.sh ssh <user@host> "
    "&& bash ladders/experiments/S1/S-R2/run.sh"
)


@dataclass(frozen=True)
class Shape:
    """One serving problem: the model, and the batch/prompt/generation the arms are compared at."""

    model: str
    batch: int
    prompt: int
    gen: int

    @property
    def label(self) -> str:
        return f"{self.model}·B{self.batch}·p{self.prompt}·g{self.gen}"


#: The shape S-R2 is measured at. Qwen3-8B at batch 32 is the plan's §05 S-R1 row ("≤ 2× at B=32"),
#: carried forward unchanged so the FP8 arm, the bf16 floor, and S-R1's vLLM comparison are all the
#: same problem. S-R1's spec.md has not declared its shape yet; this is the declaration, and S-R1
#: must match it or the two rungs stop being commensurable.
SHAPES: dict[str, Shape] = {
    "qwen3_8b_b32": Shape(model="Qwen/Qwen3-8B", batch=32, prompt=1024, gen=128),
}

ARMS = ("bf16", "fp8")

#: The engine S-R2 quantizes. It is S-R1's deliverable, not this rung's — see
#: :func:`resolve_engine_factory` for why nothing here will invent one. S-R1 is building the
#: pieces (`serving/continuous.py`, `serving/loadgen.ScratchEngineAdapter`); what is missing is a
#: factory that hands back one arm with the *real* Qwen3-8B weights loaded — S-R1's own floor
#: path runs `--load-format dummy`, and a perplexity measured on dummy weights is not a number.
DEFAULT_ENGINE = "scratch_llm.serving.s_r1_engine:build_engine"


@runtime_checkable
class ServingEngine(Protocol):
    """What the runner needs from an engine. Three measurements, one arm.

    ``decode_batch`` runs one full ``batch × gen`` decode and returns the number of tokens
    generated; the runner times it and never trusts an engine's own clock. ``score_nll`` returns
    ``(summed nats, transitions scored)`` on a held-out stream. ``gsm8k_correct`` returns one flag
    per item of the named slice, **in slice order**, so the two arms can be paired item by item.
    """

    arm: str

    def decode_batch(self, shape: Shape) -> int: ...

    def score_nll(self, corpus: str) -> tuple[float, int]: ...

    def gsm8k_correct(self, slice_id: str) -> Sequence[bool]: ...


EngineFactory = Callable[[str, Shape], ServingEngine]


def resolve_engine_factory(spec: str) -> EngineFactory:
    """Import ``module:factory`` and return it. Raises with the S-R1 pointer if it is not there.

    S-R2's floor is *the bf16 engine at matched model and shape*. There is exactly one such engine
    — the one S-R1 builds — and this file will not stand in a substitute for it. A speedup measured
    against a different engine, or against a toy model with the right shape and the wrong weights,
    is a comparison to nothing (workspace invariant 2), and it would be indistinguishable from a
    real number in the ledger.
    """
    if ":" not in spec:
        raise ValueError(f"--engine must be 'module.path:factory', got {spec!r}")
    module_name, attr = spec.split(":", 1)
    try:
        module = __import__(module_name, fromlist=[attr])
    except ImportError as exc:
        raise RuntimeError(
            f"cannot import {module_name!r}: {exc}\n"
            "  S-R2 measures FP8 against S-R1's bf16 engine at matched model and shape. That\n"
            "  engine is S-R1's deliverable; nothing here fabricates one, because a floor measured\n"
            "  against a different engine is not a floor. Point --engine at it once S-R1 lands:\n"
            "    --engine my.module:build_engine     with build_engine(arm, shape) -> ServingEngine\n"
            f"    arm is one of {ARMS}; the 'fp8' arm is the one that applies\n"
            "    quant.fp8_weights.convert_linears_to_fp8 and quant.fp8_kv_per_head.PerHeadFp8KVCache."
        ) from exc
    factory = getattr(module, attr, None)
    if factory is None:
        raise RuntimeError(f"{module_name!r} has no attribute {attr!r}")
    return factory


def _time_tokens_per_s(engine: ServingEngine, shape: Shape, warmup: int, iters: int) -> dict:
    """Median tokens/s of one arm, with the p20–p80 spread that says whether to trust it."""
    sys.path.insert(0, str(_BENCH_DIR))
    from _harness import spread_pct, wallclock_ms  # noqa: PLC0415 - bench/ is not a package

    tokens: list[int] = []

    def _run() -> None:
        tokens.append(engine.decode_batch(shape))

    med_ms, lo_ms, hi_ms = wallclock_ms(_run, warmup=warmup, iters=iters)
    if not tokens:
        raise RuntimeError("engine.decode_batch was never called — wallclock_ms is broken")
    n_tokens = tokens[-1]
    if n_tokens != shape.batch * shape.gen:
        raise RuntimeError(
            f"engine produced {n_tokens} tokens, shape says {shape.batch * shape.gen} — the two "
            "arms must generate the same number of tokens or the ratio is not a speedup"
        )
    return {
        "arm": engine.arm,
        "tokens_per_s": n_tokens / (med_ms * 1e-3),
        "median_ms": med_ms,
        "p20_ms": lo_ms,
        "p80_ms": hi_ms,
        "spread_pct": spread_pct(med_ms, lo_ms, hi_ms),
        "tokens": n_tokens,
    }


def collect_evidence(
    bf16: ServingEngine, fp8: ServingEngine, *, corpus: str, gsm8k_slice: str
) -> QualityEvidence:
    """Run both arms over the same corpus and the same 200 items, paired."""
    bf16_nll, bf16_n = bf16.score_nll(corpus)
    fp8_nll, fp8_n = fp8.score_nll(corpus)
    if bf16_n != fp8_n:
        raise RuntimeError(
            f"the arms scored different token counts ({bf16_n} vs {fp8_n}) — perplexity is only "
            "comparable over the identical stream"
        )
    return QualityEvidence.from_paired(
        corpus=corpus,
        n_ppl_tokens=bf16_n,
        bf16_nll_nats=bf16_nll,
        fp8_nll_nats=fp8_nll,
        gsm8k_slice=gsm8k_slice,
        bf16_correct=bf16.gsm8k_correct(gsm8k_slice),
        fp8_correct=fp8.gsm8k_correct(gsm8k_slice),
    )


def evidence_report(ev: QualityEvidence, v: str) -> str:
    """The evidence, laid out so the gate's verdict can be argued with rather than believed."""
    b, c = ev.discordant
    return "\n".join(
        [
            f"corpus {ev.corpus} · {ev.n_ppl_tokens} transitions",
            f"  ppl        bf16 {ev.bf16_ppl:.6f}   fp8 {ev.fp8_ppl:.6f}   "
            f"rel increase {ev.ppl_rel_increase:+.6e}",
            f"gsm8k {ev.gsm8k_slice} · {ev.n_items} items, paired",
            f"  accuracy   bf16 {ev.bf16_accuracy:.4f}   fp8 {ev.fp8_accuracy:.4f}   "
            f"drop {ev.accuracy_drop:+.4f}",
            f"  discordant bf16-only-correct b={b}   fp8-only-correct c={c}   "
            f"informative items b+c={b + c}",
            f"verdict    {v}",
        ]
    )


def _no_cuda_exit() -> None:
    import torch

    if torch.cuda.is_available():
        return
    print(
        "s1_fp8_serving: no CUDA device — this rung's number is a serving measurement and there is\n"
        "  no silicon here. On the box:\n"
        f"    {BOX}",
        file=sys.stderr,
    )
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rung", choices=["S-R2"], help="measure the FP8 arm against the bf16 floor")
    mode.add_argument("--floor", choices=["bf16"], help="measure the floor alone")
    ap.add_argument("--shape", default="qwen3_8b_b32", choices=sorted(SHAPES))
    ap.add_argument("--engine", default=DEFAULT_ENGINE, help="module.path:factory(arm, shape)")
    ap.add_argument("--corpus", default="wikitext2-test", help="held-out stream both arms score")
    ap.add_argument("--gsm8k-slice", default="gsm8k-test[:200]", help="the paired 200-item slice")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="write the full row here")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, measure nothing")
    ns = ap.parse_args(argv)

    shape = SHAPES[ns.shape]
    plan = {
        "rung": "S1/S-R2",
        "mode": "floor" if ns.floor else "rung",
        "shape": ns.shape,
        "problem": asdict(shape),
        "engine": ns.engine,
        "corpus": ns.corpus,
        "gsm8k_slice": ns.gsm8k_slice,
        "warmup": ns.warmup,
        "iters": ns.iters,
        "seed": ns.seed,
    }
    if ns.dry_run:
        print(json.dumps(plan, indent=1))
        print(f"# would measure: {shape.label} · arms {ARMS if ns.rung else ('bf16',)}")
        print("# the rung's number is gated on quant.s_r2_gate.verdict(...) == PASS")
        return 0

    _no_cuda_exit()
    import torch

    torch.manual_seed(ns.seed)
    # Which FP8 GEMM actually ran. The rowwise path applies both scales inside the kernel; the
    # fallback runs the GEMM with unit scales into fp32 and pays an extra pass over the output to
    # apply them. That is a real difference in the fp8 arm's time, and without it in the row a
    # disappointing speedup cannot be attributed (spec's Wall line: "a dequant epilogue that was
    # not fused").
    plan["scaled_mm_path"] = (
        "rowwise" if rowwise_scaled_mm_supported(torch.cuda.current_device()) else "unfused"
    )
    plan["use_fast_accum"] = DEFAULT_FAST_ACCUM
    print(f"# fp8 gemm path: {plan['scaled_mm_path']} · use_fast_accum={DEFAULT_FAST_ACCUM}")
    factory = resolve_engine_factory(ns.engine)

    sys.path.insert(0, str(_BENCH_DIR))
    from _harness import provenance_line  # noqa: PLC0415 - bench/ is not a package

    print(provenance_line(f"S1/S-R2 {shape.label}"))

    bf16_engine = factory("bf16", shape)
    bf16_row = _time_tokens_per_s(bf16_engine, shape, ns.warmup, ns.iters)
    print(
        f"bf16 floor   {bf16_row['tokens_per_s']:.2f} tok/s "
        f"(median {bf16_row['median_ms']:.2f} ms, spread {bf16_row['spread_pct']:.1f}%)"
    )

    if ns.floor:
        if ns.json:
            ns.json.parent.mkdir(parents=True, exist_ok=True)
            ns.json.write_text(json.dumps({**plan, "bf16": bf16_row}, indent=1))
        print(f"{bf16_row['tokens_per_s']:.4f}")
        return 0

    fp8_engine = factory("fp8", shape)
    fp8_row = _time_tokens_per_s(fp8_engine, shape, ns.warmup, ns.iters)
    speedup = fp8_row["tokens_per_s"] / bf16_row["tokens_per_s"]
    print(
        f"fp8  rung    {fp8_row['tokens_per_s']:.2f} tok/s "
        f"(median {fp8_row['median_ms']:.2f} ms, spread {fp8_row['spread_pct']:.1f}%)"
    )

    ev = collect_evidence(bf16_engine, fp8_engine, corpus=ns.corpus, gsm8k_slice=ns.gsm8k_slice)
    try:
        thresholds = gate_thresholds(ev)
    except NotImplementedError as exc:
        print(f"s1_fp8_serving: {exc}", file=sys.stderr)
        print(
            "  The speedup is measured and is NOT printed: an ungated speedup is a property of the\n"
            "  number format, not a result. Fill quant/s_r2_gate.gate_thresholds and re-run.",
            file=sys.stderr,
        )
        b, c = ev.discordant
        print(evidence_report(ev, "UNJUDGED (thresholds unwritten)"), file=sys.stderr)
        _maybe_write_json(ns.json, plan, bf16_row, fp8_row, speedup, ev, "UNJUDGED", (b, c))
        return 4

    v = verdict(ev, thresholds)
    print(evidence_report(ev, v))
    _maybe_write_json(ns.json, plan, bf16_row, fp8_row, speedup, ev, v, ev.discordant)
    if v != "PASS":
        print(
            f"s1_fp8_serving: gate {v} — no number. A speedup that costs quality nobody bounded is\n"
            "  not this rung's exit; write the divergence into spec.md's Result block instead.",
            file=sys.stderr,
        )
        return 3
    print(f"{speedup:.4f}")
    return 0


def _maybe_write_json(
    path: Path | None,
    plan: dict,
    bf16_row: dict,
    fp8_row: dict,
    speedup: float,
    ev: QualityEvidence,
    v: str,
    discordant: tuple[int, int],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                **plan,
                "bf16": bf16_row,
                "fp8": fp8_row,
                "speedup_vs_bf16": speedup,
                "quality": {
                    "corpus": ev.corpus,
                    "n_ppl_tokens": ev.n_ppl_tokens,
                    "bf16_ppl": ev.bf16_ppl,
                    "fp8_ppl": ev.fp8_ppl,
                    "ppl_rel_increase": ev.ppl_rel_increase,
                    "gsm8k_slice": ev.gsm8k_slice,
                    "n_items": ev.n_items,
                    "bf16_accuracy": ev.bf16_accuracy,
                    "fp8_accuracy": ev.fp8_accuracy,
                    "accuracy_drop": ev.accuracy_drop,
                    "discordant_bf16_only": discordant[0],
                    "discordant_fp8_only": discordant[1],
                },
                "verdict": v,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
