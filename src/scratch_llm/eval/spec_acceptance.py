"""F3 — de-confounded speculative-decoding acceptance, measured BY PROMPT DOMAIN.

The earlier serving measurement ran the n-gram drafter against RANDOM-weight targets and saw ~0
acceptance everywhere — that falsified nothing about domains: a random target's greedy
continuation has no structure for any drafter to track. This harness re-poses the question on a
TRAINED checkpoint (the F1 run): does prompt-lookup acceptance follow the structure of the prompt
domain (code/JSON ≫ prose), as the prompt-lookup literature claims?

Pre-registered falsifier (bench/RESULTS.md §Frontier ablations · F3; GPU day, on a trained ckpt
with val_bpb ≪ log2 V): n-gram(n=3, k=4) acceptance prose <10% (predict 5–8%), code/JSON 40–60%.
KILL: prose ≥ code/JSON, or every domain >30%.

Key invariants:
- **Consumer of the serving PUBLIC API only** (zone rule): the ``Drafter`` protocol,
  ``NGramDrafter`` and ``speculative_generate``; acceptance is read from the ``SpecStats`` the
  engine already returns, never from private serving state.
- **Losslessness rides along, never assumed:** every prompt is replayed through plain greedy
  ``sampling.generate`` and committed == greedy is reported per prompt — a broken accept/rollback
  path would silently inflate acceptance, so the oracle travels with the number.
- **Pure and composable:** model + prompts + drafter in, :class:`AcceptanceReport` out; no I/O,
  no globals. The CLI (bench/f3_deconfound_acceptance.py) owns checkpoints, tokenizers and JSON.

Interview question this answers: "your speculative-decode A/B shows ~0% acceptance — name the
confounders before you blame the drafter." (An untrained target, an unstructured prompt domain,
and an unverified accept path — this harness isolates all three.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scratch_llm.eval.protocols import TextTokenizer
from scratch_llm.model import TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.speculative import Drafter, SpecStats, speculative_generate


@dataclass(frozen=True)
class DomainPrompt:
    """One prompt tagged with its domain label. ``ids`` is the tokenized prompt; ``text`` is the
    optional source string it came from (kept for the JSON artifact, never re-tokenized)."""

    domain: str
    ids: tuple[int, ...]
    text: str | None = None


@dataclass(frozen=True)
class PromptResult:
    """One prompt's measurement: the speculative engine's accounting plus the greedy oracle."""

    prompt: DomainPrompt
    committed_ids: tuple[int, ...]  # what speculative_generate committed
    greedy_ids: tuple[int, ...]  # what plain greedy `generate` produces — the oracle
    stats: SpecStats

    @property
    def lossless(self) -> bool:
        """Exact token identity — the speculative-decode correctness contract."""
        return self.committed_ids == self.greedy_ids

    @property
    def token_agreement(self) -> float:
        """Fraction of positions where committed == greedy (diagnostic for fp32/bf16 tie-flips;
        1.0 whenever :attr:`lossless`). Length mismatch counts as disagreement."""
        n = max(len(self.committed_ids), len(self.greedy_ids))
        if n == 0:
            return 1.0
        agree = sum(a == b for a, b in zip(self.committed_ids, self.greedy_ids, strict=False))
        return agree / n


@dataclass(frozen=True)
class DomainStats:
    """One domain's aggregate row — field-wise sums over its :class:`PromptResult`s."""

    domain: str
    n_prompts: int
    n_tokens: int  # committed tokens across the domain's prompts
    n_decode_forwards: int  # target forwards EXCLUDING the one prefill per prompt
    n_drafted: int
    n_accepted: int
    n_lossless: int  # prompts whose committed == greedy exactly
    mean_token_agreement: float

    @property
    def acceptance_rate(self) -> float:
        """accepted / drafted — 0.0 when nothing was drafted (the honest null, not NaN)."""
        return self.n_accepted / self.n_drafted if self.n_drafted else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """Committed tokens per decode forward — the speedup proxy (plain decode = 1.0)."""
        return self.n_tokens / self.n_decode_forwards if self.n_decode_forwards else 0.0

    @property
    def lossless(self) -> bool:
        return self.n_lossless == self.n_prompts


@dataclass(frozen=True)
class AcceptanceReport:
    """The full domain-acceptance measurement: per-domain rows (first-appearance order) plus the
    per-prompt evidence behind them, and the knobs the numbers are conditional on."""

    rows: tuple[DomainStats, ...]
    prompts: tuple[PromptResult, ...]
    k: int
    max_new_tokens: int

    @property
    def lossless(self) -> bool:
        """True iff EVERY prompt in every domain committed exactly the greedy sequence."""
        return all(row.lossless for row in self.rows)

    def by_domain(self) -> dict[str, DomainStats]:
        return {row.domain: row for row in self.rows}


def tokenize_domain_prompts(
    tokenizer: TextTokenizer,
    domains: Mapping[str, Sequence[str]],
) -> list[DomainPrompt]:
    """Tokenize ``{domain: [texts]}`` into :class:`DomainPrompt`s, preserving mapping order."""
    return [
        DomainPrompt(domain=domain, ids=tuple(tokenizer.encode(text)), text=text)
        for domain, texts in domains.items()
        for text in texts
    ]


def measure_domain_acceptance(
    model: TransformerLM,
    prompts: Sequence[DomainPrompt],
    drafter: Drafter,
    *,
    max_new_tokens: int = 64,
    k: int = 4,
    device: str = "cpu",
) -> AcceptanceReport:
    """Run greedy speculative decoding over ``prompts`` and aggregate acceptance per domain.

    For every prompt: one ``speculative_generate`` pass (acceptance accounting via the public
    ``SpecStats``) plus one plain greedy ``generate`` replay (the losslessness oracle). The
    ``drafter`` must be stateless across prompts (``NGramDrafter``/``ModelDrafter`` both are).
    Fails loud on degenerate inputs — an empty prompt set or an empty prompt would otherwise
    surface as a silently-zero acceptance row.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be ≥ 1, got {max_new_tokens}")
    if k < 0:
        raise ValueError(f"k must be ≥ 0, got {k}")
    if len(prompts) == 0:
        raise ValueError("prompts must be non-empty")
    for p in prompts:
        if len(p.ids) == 0:
            raise ValueError(f"domain {p.domain!r}: prompt ids must be non-empty")

    results: list[PromptResult] = []
    for p in prompts:
        committed, stats = speculative_generate(
            model, drafter, p.ids, max_new_tokens, device=device, k=k
        )
        greedy = generate(
            model,
            p.ids,
            SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
            device=device,
        )
        results.append(
            PromptResult(
                prompt=p,
                committed_ids=tuple(committed),
                greedy_ids=tuple(greedy),
                stats=stats,
            )
        )

    domain_order: list[str] = []
    grouped: dict[str, list[PromptResult]] = {}
    for res in results:
        if res.prompt.domain not in grouped:
            domain_order.append(res.prompt.domain)
            grouped[res.prompt.domain] = []
        grouped[res.prompt.domain].append(res)

    rows = tuple(
        DomainStats(
            domain=domain,
            n_prompts=len(group),
            n_tokens=sum(r.stats.n_tokens for r in group),
            n_decode_forwards=sum(r.stats.n_target_forwards - 1 for r in group),
            n_drafted=sum(r.stats.n_drafted for r in group),
            n_accepted=sum(r.stats.n_accepted for r in group),
            n_lossless=sum(r.lossless for r in group),
            mean_token_agreement=sum(r.token_agreement for r in group) / len(group),
        )
        for domain, group in ((d, grouped[d]) for d in domain_order)
    )
    return AcceptanceReport(rows=rows, prompts=tuple(results), k=k, max_new_tokens=max_new_tokens)


def report_to_markdown(report: AcceptanceReport) -> str:
    """Render the per-domain rows as a markdown table (the bench ledger fragment)."""
    lines = [
        "| domain | prompts | drafted | accepted | acceptance | tok/decode-fwd | lossless |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {row.domain} | {row.n_prompts} | {row.n_drafted} | {row.n_accepted} "
        f"| {row.acceptance_rate:.1%} | {row.tokens_per_forward:.2f} "
        f"| {row.n_lossless}/{row.n_prompts} |"
        for row in report.rows
    )
    return "\n".join(lines)


__all__ = [
    "AcceptanceReport",
    "DomainPrompt",
    "DomainStats",
    "PromptResult",
    "measure_domain_acceptance",
    "report_to_markdown",
    "tokenize_domain_prompts",
]
