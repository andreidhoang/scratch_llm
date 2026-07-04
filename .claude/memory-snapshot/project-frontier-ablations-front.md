---
name: project-frontier-ablations-front
description: "2026-07-04 — third front (ADR-0018) on scratch_llm: close the loop (nanochat spine) + EV-ranked ablation study. LOOP CLOSES — F1 Muon + train-wiring/F4 + eval report card + speedrun spine SHIPPED/pushed (25ac101→391bad9); GPU-verified talking sample. Next = F1 iso-FLOP run → F2 MTP"
metadata: 
  node_type: memory
  type: project
  originSessionId: fd837ef7-225c-46e9-a277-db48c02a1c67
---

The **close-the-loop / frontier-ablation front** (2026-07-04, ADR-0018) — a THIRD parallel front on
`/workspace/scratch_llm`, alongside [[project-perf-curriculum-sprint]] and the CS336/DELTA work
([[project-cs336-delivery-sprint]]). Thesis pivot: CS336 built every *layer* but never ran the
*loop* — no model trained end-to-end into something that talks (everything real is rental-gated;
serving numbers were measured on random-weight toys, `bench/RESULTS.md:383`). Adopt Karpathy
**nanochat**'s `speedrun.sh` spine + report card to train a real talking model (headline = **nanochat
d20 ≈561M, ≈11–12B tokens, ~$100 8×H100**; user chose straight-to-rental), then run an EV-ranked,
pre-registered, iso-FLOP **ablation study**.

**Source of truth:** `docs/FRONTIER_2026_ABLATIONS.md` (spec + execution DAG F1..F9, model tiers,
verification, honesty ledger) · `docs/adr/ADR-0018` · ledger `bench/RESULTS.md` §Frontier ablations ·
plan of record `~/.claude/plans/misty-sniffing-cerf.md`. Ranked 80/20 = **F1 Muon · F2 MTP draft
head · F3 de-confound serving · F4 bf16+compile**, then F5 MLA-for-real · F6 MoE-balancing · F7 GRPO
"aha" · F8 DSA · F9 logit-guard. Grounded in research pass `wjzztlnz7` (54 CONFIRMED / 1 REFUTED /
4 UNCERTAIN vs primary sources: Muon/Moonlight 2502.16982, Kimi-K2 2507.20534, DeepSeek V2/V3/R1/V3.2,
GLM-5.2 = the user's "GLM5-2", real as of 2026-06-16).

**Progress — THE LOOP CLOSES, 2026-07-04 (all pushed to origin/main, full green-CI + GPU-verified):**
- **F1 Muon** (25ac101): `Muon` (NS5 + Nesterov 0.95 + Moonlight RMS-match `0.2·√max(A,B)`) +
  `split_muon_adamw_params` in `optim.py`. Honest: 2 NS over-claims falsified+corrected (NS5
  compresses σ→band ~[0.68,1.14], not "all→1"; median≈0.77).
- **F1/F4 train wiring** (4d859a6): `CombinedOptimizer` + `build_optimizer` + `TrainConfig`
  {optimizer, amp_dtype, compile} in `train.py` + a loud non-finite-loss guard. GPU: muon_adamw
  learns to 8e-4 (fp32/bf16/compile each); **`[FACT]` bf16+compile NaN on sm120/torch-2.12 inductor**
  (repro's with plain AdamW — inductor bug, not our logic; caught by the guard; H100 is the path).
- **F-front eval** (9e61d7a): `src/scratch_llm/eval/` — `val_bpb` + MC (ARC/MMLU) + generative
  (GSM8K/HumanEval) + CORE-style aggregate; 8 tests.
- **Phase-0 speedrun — LOOP CLOSES** (391bad9): `scratch_llm/speedrun.py` + `scripts/speedrun.sh`
  = tokenizer→pretrain(MuonAdamW)→eval→sample, one `--depth` knob (20 ⇒ d20 headline). GPU nano run
  8.3 s: val_bpb 0.02 + a **coherent talking sample** — the first model this repo can sample from.
  `--nano` = the CPU pre-flight before the $100 d20 rental.

**Buildable next-phase DAG (23 rungs, code-grounded by workflow w77bbp4pb):**
`docs/FRONTIER_2026_TASKSPEC.md` — Track A (loop→chat: A1 real shards · A0 decontam · A2 ckpt-chain ·
A3 chat template · A4 midtrain · A5 SFT · A6 chat REPL · A7 distributed d20 · A8 rental runbook+guardrails ·
A9 model card) + Track B (F1-run iso-FLOP headline · F2a/b MTP · F3 de-confound · F5 MLA-real · F6 MoE ·
F7a/b/c GRPO aha · F9 MuonClip · F8 DSA). Each rung has exact interfaces + tests + falsifier + kill + zone.

**▶ NEXT NODE = A1** (real-corpus FineWeb shards, `data/shards.py`) → **A2** (ckpt chaining) →
**F1-run** (the still-PENDING iso-FLOP Muon-vs-AdamW headline). Tracked as tasks; each rung:
pre-register falsifier in RESULTS.md → test-first → green-CI → commit → push.

**Fresh-session pickup is WIRED (commit 49bd982):** the SessionStart hook (`.claude/hooks/session-start.sh`)
greps a `Next-node:` marker from `FRONTIER_2026_TASKSPEC.md` and injects `frontier node → A1…` into every
new session; TASKSPEC has a START-HERE block; STATUS.md + CLAUDE.md point at it. **On shipping a rung,
advance the `Next-node:` HTML-comment marker at the top of FRONTIER_2026_TASKSPEC.md** so the next session
picks up the right rung. Zone: NEVER edit perf-owned mla.py/serving/kernels/quant — satisfy their Protocols.

**Concurrency `[FACT]` (learned 2026-07-04):** three fronts share ONE checkout and commit to `main`
rapidly. A concurrent front's broad `git add` swept my uncommitted `tests/test_optim.py` into ITS
commit *without* my `optim.py`, leaving HEAD red (test imported a `Muon` class not yet committed) —
fixed by committing the impl. Mitigation: commit impl+tests together promptly; precise `git add`
(never `-A`); pull-rebase before push; expect the tree to be transiently red from another front's WIP.

**MODEL mastery-derivation roadmap SHIPPED (2026-07-04, commit 813faa0):** `docs/learning/roadmap_model/`
— the twin of the perf `docs/learning/roadmap/`, covering the OTHER half (how the model LEARNS). 10 série /
48 Bài (VI, derivation-first, pinned 4ad0ac5, anchors spot-verified), heavily cross-referenced to the
vendored frontier impls (`.venv/.../transformers/models/{deepseek_v2,deepseek_v3,deepseek_v32,glm4_moe,
nanochat,qwen3}`): M1 tokenizer · M2 transformer · M3 objective+optimization (Muon/Newton-Schulz, honesty-
ledger RMS=1/√max carried in) · M4 training loop (F4 bf16/compile NaN finding) · M5 distributed (DDP→ZeRO→
FSDP) · M6 scaling laws · M7 data · M8 post-training/RL (SFT→GRPO/Dr.GRPO→DPO, densest 39KB) · M9 MoE·MLA·
MTP · M10 close-the-loop + F1–F9 as derivation. Honesty: every Neo typed (measured invariant / real
RESULTS.md number / labeled PREDICTION — never fabricated; most training runs rental-gated). Built by a
10-agent Workflow. INDEX.md leads with both roadmaps (model first); CLAUDE.md "Where things live" updated.
This is the study map for the F1–F9 ablations + the whole CS336 model stack.

**MASTER CURRICULUM merging BOTH roadmaps SHIPPED (2026-07-04, commit 0742686):** `docs/learning/CURRICULUM.md`
— one ordered mastery path unifying MODEL (M1–M10) + PERF (S1–S6) = 8 stages / 16 série / **89 Bài** in
dependency order for a senior AI RE: make-it-work (M1–M4) → make-it-fast/kernels (S3–S4) → scale (M5+S6) →
science+data (M6–M7) → align (M8) → frontier arch (M9) → serve (S1–S2,S5) → prove/ablations (M10 capstone).
Interleaves at the joints (kernels after the forward pass; serving after MoE/MLA/MTP; roofline enters twice
— S3.0 kernel-level, S1.0 decode-level). `docs/learning/INDEX.md` + CLAUDE.md "Where things live" now LEAD
with CURRICULUM.md as the single entry point. Read order for the whole repo's mastery track starts here.
