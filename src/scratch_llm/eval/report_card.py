"""The report card — the loop's acceptance oracle (docs/FRONTIER_2026_ABLATIONS.md §7).

Bundles the intrinsic metric (``val_bpb``) with the task families (multiple-choice, generative) into
one comparable artifact, plus a **CORE-style** aggregate. Honest naming: ``core_style_score`` is the
DCLM-CORE *aggregation recipe* (random-baseline-centered mean accuracy) — it is the official CORE
score only when fed the official CORE task suite; on an arbitrary task set it is a CORE-*style*
number, not the leaderboard metric.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from scratch_llm.eval.generative import GenResult, evaluate_generative
from scratch_llm.eval.metrics import BpbResult, bits_per_byte
from scratch_llm.eval.multiple_choice import MCResult, evaluate_multiple_choice
from scratch_llm.eval.protocols import TextTokenizer
from scratch_llm.model import TransformerLM
from scratch_llm.sampling import SamplingParams


@dataclass(frozen=True)
class MCTask:
    name: str
    examples: Sequence[tuple[str, Sequence[str], int]]  # (prompt, options, correct_index)
    random_baseline: float  # 1 / n_options — the CORE-style centering point


@dataclass(frozen=True)
class GenTask:
    name: str
    examples: Sequence[tuple[str, str]]  # (prompt, reference)
    grade_fn: Callable[[str, str], bool]


def core_style_score(task_accuracy: dict[str, float], baselines: dict[str, float]) -> float:
    """CORE-style aggregate: the mean over tasks of the random-baseline-centered accuracy
    ``(acc − base) / (1 − base)`` (0 = chance, 1 = perfect). Empty → 0.0."""
    centered = [
        (acc - baselines.get(name, 0.0)) / (1 - baselines.get(name, 0.0))
        for name, acc in task_accuracy.items()
        if baselines.get(name, 0.0) < 1.0
    ]
    return sum(centered) / len(centered) if centered else 0.0


@dataclass(frozen=True)
class ReportCard:
    val_bpb: float | None = None
    nats_per_token: float | None = None
    mc: dict[str, float] = field(default_factory=dict)
    generative: dict[str, float] = field(default_factory=dict)
    core_style: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "val_bpb": self.val_bpb,
            "nats_per_token": self.nats_per_token,
            "mc": dict(self.mc),
            "generative": dict(self.generative),
            "core_style": self.core_style,
        }

    def to_markdown(self) -> str:
        lines = ["| metric | value |", "|---|---|"]
        if self.val_bpb is not None:
            lines.append(f"| val_bpb | {self.val_bpb:.4f} |")
        if self.nats_per_token is not None:
            lines.append(f"| nats/token | {self.nats_per_token:.4f} |")
        for name, acc in self.mc.items():
            lines.append(f"| MC · {name} | {acc:.3f} |")
        for name, acc in self.generative.items():
            lines.append(f"| GEN · {name} | {acc:.3f} |")
        if self.core_style is not None:
            lines.append(f"| **CORE-style** | **{self.core_style:.4f}** |")
        return "\n".join(lines)


def build_report_card(
    model: TransformerLM,
    tokenizer: TextTokenizer,
    *,
    val_tokens: Sequence[int] | None = None,
    val_num_bytes: int | None = None,
    mc_tasks: Sequence[MCTask] = (),
    gen_tasks: Sequence[GenTask] = (),
    sampling_params: SamplingParams | None = None,
    device: str = "cpu",
    context_length: int | None = None,
) -> ReportCard:
    """Run every requested metric and assemble a :class:`ReportCard`.

    ``val_tokens`` + ``val_num_bytes`` drive ``val_bpb`` (skip both to omit it). ``mc_tasks`` /
    ``gen_tasks`` are the ARC/MMLU and GSM8K/HumanEval families; the MC tasks' accuracies feed the
    CORE-style aggregate (centered on each task's ``random_baseline``).
    """
    bpb: BpbResult | None = None
    if val_tokens is not None and val_num_bytes is not None:
        bpb = bits_per_byte(
            model, val_tokens, val_num_bytes, context_length=context_length, device=device
        )

    mc: dict[str, float] = {}
    baselines: dict[str, float] = {}
    for task in mc_tasks:
        res: MCResult = evaluate_multiple_choice(model, tokenizer, task.examples, device=device)
        mc[task.name] = res.accuracy
        baselines[task.name] = task.random_baseline

    gen: dict[str, float] = {}
    for task in gen_tasks:
        gres: GenResult = evaluate_generative(
            model, tokenizer, task.examples, task.grade_fn, params=sampling_params, device=device
        )
        gen[task.name] = gres.accuracy

    return ReportCard(
        val_bpb=None if bpb is None else bpb.bits_per_byte,
        nats_per_token=None if bpb is None else bpb.nats_per_token,
        mc=mc,
        generative=gen,
        core_style=core_style_score(mc, baselines) if mc else None,
    )
