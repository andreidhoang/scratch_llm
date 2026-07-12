# reference/ — external inference research (dated, subordinate to canon)

> **What this is.** Imported external research on the 2026 inference frontier, parked here as **on-demand
> reference**. It is *not* repo canon and does **not** create a new source of truth. It lives at the
> `cs336/` workspace root (a sibling of [`../_archive/`](../_archive/)), **outside the `scratch_llm/` git
> repo**, so it does not touch that repo's tracked doc-to-code ratio or its doc-freeze (see
> [`../STRATEGY.md`](../STRATEGY.md) §9). Added 2026-06-22.

## Contents

| File | What it is |
|---|---|
| [`Frontier_Inference_Engineering_Curriculum_2026.md`](Frontier_Inference_Engineering_Curriculum_2026.md) | A first-principles, compute-tiered inference-engineering curriculum + capstone ladder (general learning roadmap). |
| [`Frontier_Inference_2026_Research_Brief.md`](Frontier_Inference_2026_Research_Brief.md) | The cited 2026 landscape brief (serving systems · algorithms · hardware/kernels · frontier models · talent) — the curriculum's evidence base. |

## The one rule (so this stays a reference, not a second plan)

**The repo's canon wins.** Where this material and the repo disagree on *this project's* direction, the
repo's documents are authoritative. This folder is breadth/landscape context and an interview-prep reading
layer — use it to go wider, then act from canon.

### Canonical inference material in this repo (go here to *act*)

- **Plan of record — the inference/kernel capstone:** [`../DELTA.md`](../DELTA.md) (GDN-2 decode-kernel
  RFC + 4-week plan), tracked in [`../scratch_llm/docs/STATUS.md`](../scratch_llm/docs/STATUS.md) and
  [`../scratch_llm/docs/IMPLEMENTATION_PLAN.md`](../scratch_llm/docs/IMPLEMENTATION_PLAN.md) §7.
- **Serving-stack OSS lane + role targets:** [`../STRATEGY.md`](../STRATEGY.md) §6–§7.
- **The 2026 frontier layer (per-pillar 🟢/🔵/⚪ + the DELTA inference axis):**
  [`../scratch_llm/docs/FRONTIER_PRACTICE_2026.md`](../scratch_llm/docs/FRONTIER_PRACTICE_2026.md).
- **The inference *plan specs* (design SPECs):**
  [`L2_flash_attention_SPEC.md`](../scratch_llm/docs/design/L2_flash_attention_SPEC.md) ·
  [`L2_kv_cache_SPEC.md`](../scratch_llm/docs/design/L2_kv_cache_SPEC.md) ·
  [`L2_rollout_seam_SPEC.md`](../scratch_llm/docs/design/L2_rollout_seam_SPEC.md) ·
  [`L2_kl_train_infer_SPEC.md`](../scratch_llm/docs/design/L2_kl_train_infer_SPEC.md).
- **Lecture → code map (L10 = Inference):** [`../scratch_llm/docs/LECTURE_MAP.md`](../scratch_llm/docs/LECTURE_MAP.md).
- **The context-engineering field manual:** [`../scratch_llm/docs/CONTEXT_ENGINEERING.md`](../scratch_llm/docs/CONTEXT_ENGINEERING.md).
- **The prior inference-vs-frontier snapshot:** [`../INFERENCE_FRONTIER_2026.html`](../INFERENCE_FRONTIER_2026.html).

## Overlap map — what is *additive* vs *already covered*

So a reader knows when to use this folder and when to use canon:

- **Already in canon (use canon):** FA-3/FlashInfer, NVFP4/FP8, EAGLE-3/MTP, PagedAttention/RadixAttention,
  DistServe/Mooncake/Dynamo disaggregation, DeepSeek large-EP, the "decode is memory-bound / 100% MFU is an
  anti-goal" framing, the vLLM/SGLang PR ladder — all in `STRATEGY.md` §6 and
  `FRONTIER_PRACTICE_2026.md` (A2 + the DELTA section).
- **Additive here (the reason to keep this):** a fuller **serving-framework head-to-head** with numbers, the
  **prefill/decode disaggregation + KV-systems** map, the **Hopper→Blackwell→Rubin** hardware roadmap and
  competitor table, the **frontier-model architecture table** (DeepSeek/Llama4/Qwen3/Kimi K2/gpt-oss…), the
  **agentic/$-per-task** cost shift, and the **hiring take-home** specifics ("make this kernel faster"). Pull
  these into an interview-prep pass or a future `FRONTIER_PRACTICE_2026.md` delta — don't fork a second plan.

*Provenance: produced 2026-06-22 by a five-agent web-research sweep (serving / algorithms / hardware /
models / talent), each adversarially fact-checked. Roadmap items (Rubin, HBM4E, unreleased models) are
flagged in-file as announced-not-shipping; re-verify any number before quoting it live.*
