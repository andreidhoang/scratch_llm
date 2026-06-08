# ADR-0009 — reasoningLLM_scratch is the artifact; the older reasoningLLM repo is a design reference

- **Status:** Accepted (2026-06-08)
- **Layer:** Project strategy (repo topology) — supersedes the prior "rebuild = no" decision
- **Decides:** which of two overlapping RL repos is the one built, shipped, and shown

## Context

Two repos hold an overlapping verifier-exploitability RL stack:

- **`Desktop/reasoningLLM/`** (older): real `algos/rewards/envs/rollout`, ADRs to ADR-0010, a
  verifier-exploitability study. Audit (2026-06-08): **32 of 69 `src/*.py` are ` 2.py` Finder-copy
  duplicates (46 %)**, **0 CI workflows**, the "test" count is inflated by vendored / nested repos
  (a nested `optim_lab` repo was just untracked), and **6 files glue around `verl`** — i.e. the RL
  is *integration*, not owned-from-scratch. **No completed VERA results/runs/plots on disk** (code +
  design only).
- **`Desktop/cs336/reasoningLLM_scratch/`** (this repo): clean, owned-from-scratch, **green-CI
  enforced** (ruff/pyright/pytest, 84 tests), L1 + L2 built with real understanding (loss-at-init,
  overfit-one-batch, `kl_train_infer` measured on a 4090).

A prior "frozen decision #3" said *"Rebuild = NO, audit-and-extend the older repo."* It predates the
clean repo proving itself, and leaving two repos to silently compete is the real waste.

## Decision

1. **`reasoningLLM_scratch` is the single public artifact** — built, shipped, shown. The pitch
   ("I own one number end-to-end, token → reward") is only credible on owned-from-scratch code, not
   on `verl` glue; and clean engineering is the hiring signal (UNIFIED §5).
2. **The older `reasoningLLM` repo is demoted to a private *design reference*.** Harvest its ADRs
   (0008–0010) and the dial/oracle/ARPO/off-policy design as **prior art** that de-risks this
   repo's ADRs; **never copy its code** (the clean-room rule, `CLAUDE.md`).
3. **The prior "frozen decision #3" (rebuild = no, extend the older repo) is RETIRED.**
4. **Scope by the from-scratch-value test:** build the owned core here — GRPO/Dr.GRPO + advantage +
   off-policy (CS336 A5 = the 45-minute interview gate) and the exploitability dial + true-quality
   oracle (the differentiator). Keep *infrastructure* minimal — **no `verl` / async-orchestrator /
   r2e_gym rebuild**; `LocalBackend` + a thin SGLang client suffice for the smoke run and a 1.5B
   VERA run.

## Consequences

- (+) One clean, owned, green-CI repo — the artifact a frontier lab screens for; no two-repo
  ambiguity.
- (+) "RL again" is not duplication: rebuilding the *algorithm* (GRPO) + dial/oracle **is** the
  CS336 mastery and the only honest way to own the finding. Only the *plumbing* would have been
  duplication, and that is explicitly not rebuilt.
- (+) Nothing expensive is lost: the older repo has **no completed VERA runs**, so demotion costs no
  GPU-hours; only design is harvested.
- (−) The older repo's verl-integrated infra is not carried forward; if real distributed/serving
  infra is later needed it is re-scoped fresh, not ported.
- **Builds on** the `docs/IMPLEMENTATION_PLAN.md` integration spine (§2 scope tiers); the build-ADR
  block there shifts to 0010–0015 to make room for this strategy ADR.
