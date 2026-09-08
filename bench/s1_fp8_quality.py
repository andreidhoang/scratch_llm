"""S1/S-R2 — the FP8 quality gate, run on HF-transformers Qwen3-8B.

The other half of ``s1_fp8_serving.py``. That file measures the speedup and gates it on a
quality verdict computed from the *same* engines; this file computes only the verdict, from
HF-transformers Qwen3-8B with our quantization scheme applied to it. Two entry points because
the two questions have different requirements: a speedup needs our engine at matched model
and shape or it is a comparison to nothing, while Δppl needs real Qwen3-8B weights, which our
own loader does not yet read.

It prints **no tok/s and no speedup** — see ``QualityArm.decode_batch``. What it prints
is the evidence and a verdict, in the same layout ``s1_fp8_serving`` uses, so the two runs are
read side by side. The verdict still comes from ``quant.s_r2_gate``: if Huy's thresholds are
unwritten this exits 4 and judges nothing, exactly as the sibling does.

    PYTHONPATH=src python bench/s1_fp8_quality.py --json results/quality.json

Scope, restated because a PASS here is narrower than it looks: this validates the
quantization *scheme*, not our engine's FP8 GEMM path. The module docstring of
``serving/quality_arms.py`` says why, and that limitation belongs in the rung's write-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scratch_llm.quant.s_r2_gate import gate_thresholds, verdict
from scratch_llm.serving.quality_arms import DEFAULT_MODEL_ID, build_quality_arms

# `scratch_llm.bench` is the installed package (src/scratch_llm/bench: harness, ledger,
# gpu_specs); THIS directory is the un-packaged runner tree and is not importable by that
# name. Siblings here reach each other through sys.path, as bench/kernels/gemm/k1_ladder.py
# does with _harness. collect_evidence is imported rather than reimplemented because it holds
# the checks that make the two arms comparable — the identical-token-count guard above all —
# and a second copy of those is a second place for them to drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_fp8_serving import collect_evidence, evidence_report  # noqa: E402

BOX = (
    "infra/rent.sh sync-up <user@host> && infra/rent.sh ssh <user@host> "
    "&& PYTHONPATH=scratch_llm/src python scratch_llm/bench/s1_fp8_quality.py"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--corpus", default="wikitext2-test")
    ap.add_argument("--gsm8k-slice", default="gsm8k-test[:200]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="write the evidence here")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    ns = ap.parse_args(argv)

    plan = {
        "rung": "S1/S-R2",
        "mode": "quality",
        "model_id": ns.model_id,
        "corpus": ns.corpus,
        "gsm8k_slice": ns.gsm8k_slice,
        "seed": ns.seed,
        "measures": "delta-ppl + paired GSM8K only; NO tok/s, NO speedup",
        "validates": "the quantization scheme, not our engine's fp8 GEMM path",
    }
    if ns.dry_run:
        print(json.dumps(plan, indent=1))
        return 0

    import torch

    if not torch.cuda.is_available():
        print(
            f"s1_fp8_quality: needs CUDA — Qwen3-8B in two arms. Run it on the box:\n  {BOX}",
            file=sys.stderr,
        )
        return 2
    torch.manual_seed(ns.seed)

    bf16_arm, fp8_arm = build_quality_arms(model_id=ns.model_id)
    plan["fp8_layers_converted"] = len(fp8_arm.converted)
    print(f"fp8 arm: {len(fp8_arm.converted)} nn.Linear layers replaced with Fp8Linear")

    ev = collect_evidence(bf16_arm, fp8_arm, corpus=ns.corpus, gsm8k_slice=ns.gsm8k_slice)
    try:
        thresholds = gate_thresholds(ev)
    except NotImplementedError as exc:
        print(f"s1_fp8_quality: {exc}", file=sys.stderr)
        print(evidence_report(ev, "UNJUDGED (thresholds unwritten)"), file=sys.stderr)
        _write(ns.json, plan, ev, "UNJUDGED")
        return 4

    v = verdict(ev, thresholds)
    print(evidence_report(ev, v))
    _write(ns.json, plan, ev, v)
    return 0 if v == "PASS" else 1


def _write(path: Path | None, plan: dict, ev, v: str) -> None:
    if path is None:
        return
    b, c = ev.discordant
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                **plan,
                "n_ppl_tokens": ev.n_ppl_tokens,
                "bf16_ppl": ev.bf16_ppl,
                "fp8_ppl": ev.fp8_ppl,
                "ppl_rel_increase": ev.ppl_rel_increase,
                "n_items": ev.n_items,
                "bf16_accuracy": ev.bf16_accuracy,
                "fp8_accuracy": ev.fp8_accuracy,
                "accuracy_drop": ev.accuracy_drop,
                "discordant_b": b,
                "discordant_c": c,
                "verdict": v,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
