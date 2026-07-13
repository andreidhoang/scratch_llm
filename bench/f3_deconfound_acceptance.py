"""F3 — de-confound n-gram speculative acceptance BY PROMPT DOMAIN, on a TRAINED checkpoint.

The random-weights serving measurement saw ~0 acceptance on every prompt — a null that says
nothing about domains (an untrained target's greedy continuation has no structure to track).
This driver re-measures on a trained ckpt (the F1 run's; val_bpb ≪ log2 V is the precondition).

Pre-registered in bench/RESULTS.md §Frontier ablations (07-04, recalibrated 07-09 BEFORE the run):
- n-gram(n=3, k=4) acceptance: prose <10% (predict 5–8%), code/JSON 40–60%.
- Losslessness preserved: committed ids == greedy `generate` per prompt (fp32 tie-flips reported
  as token-agreement, not silently absorbed).
- **KILL:** prose ≥ code/JSON, or every domain >30%.

Run (GPU day; CPU works too — the model is small):
  python bench/f3_deconfound_acceptance.py --ckpt runs/f1/model_final.pt \
      --tokenizer-dir data/fineweb_edu --device cuda --out f3_acceptance.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scratch_llm.data.shards import load_tokenizer
from scratch_llm.eval.spec_acceptance import (
    AcceptanceReport,
    measure_domain_acceptance,
    report_to_markdown,
    tokenize_domain_prompts,
)
from scratch_llm.serving.speculative import NGramDrafter
from scratch_llm.train import build_model_from_checkpoint

# Built-in domain suite: small but honest — prose avoids repetition; code/JSON carry the repeated
# identifiers/keys that prompt-lookup speculation feeds on (Saxena 2023).
DOMAIN_PROMPTS: dict[str, list[str]] = {
    "prose": [
        "The valley had been quiet for most of the winter, and the first thaw arrived without "
        "warning. Water moved under the ice long before anyone saw it move, and the old bridge "
        "groaned about it every night.",
        "History rarely announces its turning points. The treaty was signed on an ordinary "
        "Tuesday, and the newspapers gave it four lines beneath the shipping forecast.",
        "She walked the length of the platform twice before admitting the train was gone, then "
        "bought a coffee and watched the pigeons argue over crumbs near the ticket machines.",
    ],
    "code": [
        "def get_user(user_id):\n"
        "    user = db.get(user_id)\n"
        "    if user is None:\n"
        "        raise KeyError(user_id)\n"
        "    return user\n"
        "\n"
        "\n"
        "def get_order(order_id):\n"
        "    order = db.get(order_id)\n"
        "    if order is None:\n"
        "        raise KeyError(order_id)\n"
        "    return order\n"
        "\n"
        "\n"
        "def get_item(item_id):\n"
        "    item = db.get(item_id)\n",
        "parser.add_argument('--batch-size', type=int, default=32)\n"
        "parser.add_argument('--context-length', type=int, default=1024)\n"
        "parser.add_argument('--max-steps', type=int, default=1000)\n"
        "parser.add_argument('--eval-every', type=int, default=100)\n"
        "parser.add_argument('--seed', type=int, default=0)\n"
        "parser.add_argument('--device', type=str, default='cuda')\n",
        "for name in names:\n"
        "    total_by_name[name] = total_by_name.get(name, 0) + counts[name]\n"
        "for city in cities:\n"
        "    total_by_city[city] = total_by_city.get(city, 0) + counts[city]\n"
        "for team in teams:\n"
        "    total_by_team[team] = total_by_team.get(team, 0) + counts[team]\n",
    ],
    "json": [
        '{"users": [{"id": 1, "name": "alice", "active": true}, '
        '{"id": 2, "name": "bob", "active": true}, '
        '{"id": 3, "name": "carol", "active": false}, '
        '{"id": 4, "name": "dave", "active": true}, '
        '{"id": 5, "name": ',
        '{"products": [{"sku": "A-100", "price": 9.99, "stock": 12}, '
        '{"sku": "A-101", "price": 14.99, "stock": 7}, '
        '{"sku": "A-102", "price": 4.99, "stock": 31}, '
        '{"sku": "A-103", "price": ',
        '{"config": {"train": {"lr": 0.001, "steps": 1000, "batch": 32}, '
        '"eval": {"lr": 0.0, "steps": 100, "batch": 32}, '
        '"test": {"lr": 0.0, "steps": ',
    ],
}


def _verdict(report: AcceptanceReport) -> tuple[str, dict[str, float]]:
    """Apply the pre-registered kill line to the three standard domains."""
    rates = {row.domain: row.acceptance_rate for row in report.rows}
    prose, code, js = rates["prose"], rates["code"], rates["json"]
    kill = prose >= min(code, js) or min(rates.values()) > 0.30
    verdict = "KILL" if kill else "PASS"
    if not report.lossless:
        verdict += " (LOSSLESSNESS BROKEN - harness bug, fix before trusting rates)"
    return verdict, rates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt", required=True, help="config-carrying checkpoint (train.save_checkpoint)"
    )
    ap.add_argument(
        "--tokenizer-dir", required=True, help="dir holding tokenizer.json (the shard data dir)"
    )
    ap.add_argument("--ngram-n", type=int, default=3, help="n-gram drafter context length")
    ap.add_argument("--k", type=int, default=4, help="draft tokens per round")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="f3_acceptance.json")
    args = ap.parse_args()

    model, step = build_model_from_checkpoint(args.ckpt, map_location=args.device)
    model = model.to(args.device).eval()
    tokenizer = load_tokenizer(args.tokenizer_dir)
    prompts = tokenize_domain_prompts(tokenizer, DOMAIN_PROMPTS)

    report = measure_domain_acceptance(
        model,
        prompts,
        NGramDrafter(n=args.ngram_n),
        max_new_tokens=args.max_new_tokens,
        k=args.k,
        device=args.device,
    )
    verdict, rates = _verdict(report)

    print(
        f"ckpt {args.ckpt} (step {step}) · n-gram(n={args.ngram_n}, k={args.k}) "
        f"· T={args.max_new_tokens} · device {args.device}\n"
    )
    print(report_to_markdown(report))

    Path(args.out).write_text(
        json.dumps(
            {
                "args": vars(args),
                "ckpt_step": step,
                "verdict": verdict,
                "lossless": report.lossless,
                "rows": [
                    {
                        **asdict(row),
                        "acceptance_rate": row.acceptance_rate,
                        "tokens_per_forward": row.tokens_per_forward,
                    }
                    for row in report.rows
                ],
                "prompts": [
                    {
                        "domain": res.prompt.domain,
                        "text": res.prompt.text,
                        "n_prompt_tokens": len(res.prompt.ids),
                        "acceptance_rate": res.stats.acceptance_rate,
                        "tokens_per_forward": res.stats.mean_tokens_per_forward,
                        "lossless": res.lossless,
                        "token_agreement": res.token_agreement,
                        "committed_ids": list(res.committed_ids),
                        "greedy_ids": list(res.greedy_ids),
                    }
                    for res in report.prompts
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # The ledger row (paste into bench/RESULTS.md §Frontier ablations · F3 deconfound):
    n_ok = sum(row.n_lossless for row in report.rows)
    n_all = sum(row.n_prompts for row in report.rows)
    print(
        f"\n| F3 deconfound (ckpt step {step}, n={args.ngram_n} k={args.k}, "
        f"T={args.max_new_tokens}) | acceptance by domain, trained target | "
        f"prose <10% · code/JSON 40–60% | prose ≥ code/JSON or all >30% | "
        f"prose {rates['prose']:.1%} · code {rates['code']:.1%} · json {rates['json']:.1%} "
        f"· lossless {n_ok}/{n_all} → **{verdict}** |"
    )
    print(f"full per-prompt evidence → {args.out}")


if __name__ == "__main__":
    main()
