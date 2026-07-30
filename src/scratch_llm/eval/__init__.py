"""Report-card evaluation harness — the close-the-loop acceptance oracle (ADR-0018).

`val_bpb` (intrinsic) + multiple-choice (ARC/MMLU) + generative (GSM8K/HumanEval) folded into one
:class:`ReportCard`, with a documented CORE-*style* aggregate. See
`docs/FRONTIER_2026_ABLATIONS.md` §7. `optimizer_race` is the F1-run iso-FLOP A/B harness
(Muon vs LR-tuned AdamW at fixed C=6ND). `spec_acceptance` is the F3 harness — n-gram
speculative-decode acceptance measured BY PROMPT DOMAIN on a trained checkpoint, with the
committed-equals-greedy losslessness oracle riding along.
"""

from __future__ import annotations

from scratch_llm.eval.generative import GenResult, evaluate_generative
from scratch_llm.eval.metrics import BpbResult, bits_per_byte
from scratch_llm.eval.moe_ablation import (
    AblationArm,
    AblationSpec,
    Granularity,
    ablation_table,
    build_moe_config,
    evaluate_val_loss,
    router_diagnostics,
    run_moe_ablation,
    save_ablation_results,
    train_arm,
)
from scratch_llm.eval.multiple_choice import (
    MCResult,
    evaluate_multiple_choice,
    option_logprob,
    predict_choice,
)
from scratch_llm.eval.optimizer_race import (
    ArmResult,
    RaceResult,
    SweepResult,
    nats_delta_at_budget,
    run_arm,
    run_race,
    sweep_lr,
    token_saving_fraction,
    tokens_to_match,
)
from scratch_llm.eval.report_card import (
    GenTask,
    MCTask,
    ReportCard,
    build_report_card,
    core_style_score,
)
from scratch_llm.eval.spec_acceptance import (
    AcceptanceReport,
    DomainPrompt,
    DomainStats,
    PromptResult,
    measure_domain_acceptance,
    report_to_markdown,
    tokenize_domain_prompts,
)

__all__ = [
    "AblationArm",
    "AblationSpec",
    "AcceptanceReport",
    "ArmResult",
    "BpbResult",
    "Granularity",
    "ablation_table",
    "build_moe_config",
    "evaluate_val_loss",
    "router_diagnostics",
    "run_moe_ablation",
    "save_ablation_results",
    "train_arm",
    "DomainPrompt",
    "DomainStats",
    "GenResult",
    "GenTask",
    "MCResult",
    "MCTask",
    "PromptResult",
    "RaceResult",
    "ReportCard",
    "SweepResult",
    "bits_per_byte",
    "build_report_card",
    "core_style_score",
    "evaluate_generative",
    "evaluate_multiple_choice",
    "measure_domain_acceptance",
    "nats_delta_at_budget",
    "option_logprob",
    "predict_choice",
    "report_to_markdown",
    "run_arm",
    "run_race",
    "sweep_lr",
    "token_saving_fraction",
    "tokenize_domain_prompts",
    "tokens_to_match",
]
