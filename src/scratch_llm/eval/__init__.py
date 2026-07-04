"""Report-card evaluation harness — the close-the-loop acceptance oracle (ADR-0018).

`val_bpb` (intrinsic) + multiple-choice (ARC/MMLU) + generative (GSM8K/HumanEval) folded into one
:class:`ReportCard`, with a documented CORE-*style* aggregate. See
`docs/FRONTIER_2026_ABLATIONS.md` §7.
"""

from __future__ import annotations

from scratch_llm.eval.generative import GenResult, evaluate_generative
from scratch_llm.eval.metrics import BpbResult, bits_per_byte
from scratch_llm.eval.multiple_choice import (
    MCResult,
    evaluate_multiple_choice,
    option_logprob,
    predict_choice,
)
from scratch_llm.eval.report_card import (
    GenTask,
    MCTask,
    ReportCard,
    build_report_card,
    core_style_score,
)

__all__ = [
    "BpbResult",
    "GenResult",
    "GenTask",
    "MCResult",
    "MCTask",
    "ReportCard",
    "bits_per_byte",
    "build_report_card",
    "core_style_score",
    "evaluate_generative",
    "evaluate_multiple_choice",
    "option_logprob",
    "predict_choice",
]
