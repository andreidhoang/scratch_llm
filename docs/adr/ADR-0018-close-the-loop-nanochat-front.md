# ADR-0018 — Open a third front: close the loop (nanochat spine) + a frontier ablation study

**Status:** accepted · 2026-07-04 (user-directed; research pass `wjzztlnz7`)
**Scope:** repo-level program decision. Opens a new build front alongside the perf-kernel
curriculum (`performance/PERF_PLAN.md`, ADR-0014 model) and DELTA (`../../DELTA.md`). Spec:
`docs/FRONTIER_2026_ABLATIONS.md`. Plan of record: `~/.claude/plans/misty-sniffing-cerf.md`.

## Context

CS336 A1–A5 shipped (STATUS: 461 CPU tests, advanced serving stack). Read with claims-honesty
(FOP-4, *implemented ≠ measured*), the load-bearing gap is that **the loop has never closed**: no
model has been trained end-to-end into something that chats. Verified by reading the code —
`loss-at-init`/`overfit-one-batch` pass, but there is no real-corpus pretrain, no midtrain, no SFT
chat model, no CORE/ARC/MMLU/GSM8K/HumanEval report card, no RL "aha"; `mla.py` is an unwired toy,
`moe.py` has never trained on real data, `optim.py` is AdamW-only, `train.py:136` is plain fp32,
and every serving number was measured against a **random-weight toy** (the `bench/RESULTS.md:383`
n-gram-acceptance confound is the proof). Karpathy's **nanochat** is the antidote: one minimal
`speedrun.sh` that runs the entire loop to a talking model for ~$100.

A 2026 multi-agent research pass (`wjzztlnz7`: nanochat + repo deep-dive, four frontier streams —
Muon / DeepSeek / GLM-Qwen-Kimi / MTP-serving — + an adversarial verifier) established, with
**54 claims CONFIRMED / 1 REFUTED / 4 UNCERTAIN** against primary sources, that the highest-leverage
work is to (a) close the loop and (b) turn our existing frontier *toys* into *measured* ablations.

## Decision

1. **Close the loop as the top-EV node.** Adopt nanochat's end-to-end integration spine
   (`speedrun.sh` → NEW `scripts/speedrun.sh`) + a **report card** (NEW `src/scratch_llm/eval/`:
   DCLM CORE + `val_bpb` + ARC/MMLU/GSM8K/HumanEval) over the components we already own, and
   produce a trained, evaluated, chat-capable model.

2. **Model tiers (user-directed): straight to the $100 d20.** Headline artifact = nanochat-grade
   **d20 (~561M, ~11–12B tokens, 8×H100, ~$48–100)**, target CORE ≈ GPT-2 (~0.256). A **free nano
   pre-flight** (depth ~4) runs the whole pipeline first to protect the paid run. Ablation *science*
   runs at 30–300M on the standing sm120 card.

3. **A ranked, pre-registered ablation program** (`FRONTIER_2026_ABLATIONS.md` §3), 80/20 =
   **F1 MuonAdamW · F2 MTP draft head · F3 de-confound serving · F4 bf16+compile**, then F5 MLA-real,
   F6 MoE-balancing, F7 GRPO "aha", F8 DSA (stretch), F9 logit guard. Each rung is iso-FLOP,
   one-variable, with a falsifiable prediction + kill criterion, measured in `bench/RESULTS.md`.

4. **Run as a THIRD front, in parallel** (user-directed) — perf + DELTA are *not* paused. The real
   trained model becomes the substrate the perf/DELTA fronts finally measure against (F2/F5 produce
   the decode kernels they optimize; F3/F4 give DELTA a real roofline target).

5. **Adopt Muon exactly as nanochat/Moonlight do** (F1): Muon on 2D block matrices; AdamW on the
   **weight-tied embed/head tensor** (`model.py:917`), all 1-D params, and — once untied —
   embeddings/head. Reuse AdamW's LR band via the Moonlight RMS-match (`0.2·√max(A,B)`, WD 0.1).

## Justification

- **EV.** The loop is the single missing artifact a frontier lab screens for beyond "I built the
  components"; every ablation's number is uninterpretable until it runs on a real model (F3 exists
  precisely to de-confound the serving stack). Muon is the best-validated 2025 training win
  (Moonlight: ~52% of AdamW FLOPs to match loss) and changes the very run we pay for, so it leads.
- **Convergent-defaults evidence.** MTP appears in *both* DeepSeek-V3 and GLM-4.5; Muon in
  nanochat + Moonlight + Kimi-K2; aux-loss-free balancing in V3 + GLM + Qwen3 — independent labs,
  so these are load-bearing, not fashion. That is *why* F1/F2/F6 rank where they do.
- **Honesty (verifier corrections propagated).** Muon full-rank update RMS = **1/√max(A,B)** (not
  1/max(A,B)); "Muon subsumes muP" softened to "eases, not replaces" (transfers *with* muP ≤~4B);
  nanochat d26 is a passing suggestion, not a benchmarked tier. nanochat's exact per-stage token
  counts + ReLU²-vs-SwiGLU / logit-softcap / value-embedding choices are **to be confirmed against
  `nanochat/gpt.py` at build time**, not asserted.

## Consequences

- **This front's zone:** `src/scratch_llm/{eval,}`, `scripts/`, `optim.py`, `train.py`, additive
  `model.py` (MTP head + `attn='gqa'|'mla'` + untie), `mla.py` (promote), `moe.py` (ablation
  harness), the F-rung sections of `bench/RESULTS.md`. Shared files (`CLAUDE.md`, `STATUS.md`,
  `pyproject.toml`, `bench/RESULTS.md`): pull-rebase + additive + precise `git add` (never `-A`) —
  three fronts now share the checkout.
- **`tie_embeddings` default flips to `False`** (untied, nanochat-style) — a small `model.py` change
  guarded by the loss-at-init test; keeps Muon routing unambiguous. ADR-0004 (dense substrate)
  stands; this is an additive default change, recorded here.
- **Green-CI unchanged** as the commit floor; the report card is the loop's acceptance oracle;
  numeric claims stay `[FACT]` only when measured.
- **Rental gating:** d20/d26/d32 are 8×H100 rental steps (`deploy/runbooks/`-style); nano + all
  ablation science stay on the standing card.
