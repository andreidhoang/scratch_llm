# PLAN — the only plan (2026-08-26)

One page. If any document contradicts this file, this file wins. If this file goes stale,
fix THIS file — never write a second one. Everything superseded lives in `docs/archive/`
(repo) and `~/Desktop/plan/_archive_2026-08/` (history, not law).

---

## WHAT we are building

One research program with a production landing zone, then one rebuild season:

1. **E1 — the instrument + divergence map** (CPU, $0). Sequential-vs-chunked divergence of the
   gated-delta-rule recurrence (`src/scratch_llm/mastery/`): fp64 oracle (5 lines, mine) →
   pre-registered predictions → sweep over gate × dtype × chunk → public JSON + ledger rows.
2. **Measured review of vLLM PR #45819** (batch invariance for GDN_ATTN, issue #42960) — the only
   review on that thread backed by an instrument.
3. **E2 — RL-loop divergence on a production checkpoint**: rollout-vs-trainer logprob delta +
   IS-clip fraction on **Kimi-Linear-48B-A3B** (1×H200, bf16). The VeXact result for hybrids.
4. **E3 — a PR to vLLM**: the fix (or its regression harness) that the map specifies.
5. **Verifier**: `tolerances.yaml` derived only from measured envelopes — the anti-KernelBench artifact.
6. **Season 2 (after Oct 16): rebuild Kimi-K3 from scratch** (`docs/k3/ROADMAP.md`, K0–K10) on owned
   math, with E1/E2 as its numerics CI. Not before.

## WHY

- **Science:** the GDN recurrence ships in 397B production models; its two computational paths
  disagree (13% on record, FLA #389); GRPO divides their logprobs; nobody has published the map;
  MiniMax reverted a whole architecture over exactly this trust gap.
- **Market:** root-cause + measurement = requirement #1 (38/45 postings); a measured vLLM
  contribution is the strongest no-PhD credential. **Spearheads: Anthropic (Perf-RL / Perf-GPU,
  $350–850k, visa sponsored) and NVIDIA (AI Inference Perf new-grad, $124–241k, the one door with
  no hard degree gate).** Behind them: TML, Together SG (EP ≈3 wks), Cohere, xAI.
- **Personal:** frontier seat ASAP. Quantified: Anthropic-tier comp ≈ 1–1.5 years to the family
  goal; VN salary ≈ never. This window is the only ASAP-consistent path.

## HOW — the schedule (all of it)

| Week (ends) | Deliverable | $ | Hard date |
|---|---|---|---|
| **W2 (30/08)** | **E1 public** + streak broken + batch-1 applications (≥2 day-of, 13 total) + #45819 measured review + E002 cross-arch (5090 + 1h H100) | ~$10 | E1: **now** |
| W3 (06/09) | Divergence-map write-up public · **rebuild `chunked_wy` from blank** (my hand, vs my own oracle + tests → I own 100% of the instrument's math) · E2 pre-registration · reasoningLLM run | ~$40 | — |
| W4 (13/09) | Gate-distribution on real checkpoint (1×H200) · NCCL α–β: PCIe + NVLink 2×H100 | ~$85 | — |
| W5 (20/09) | **PR opened on vLLM** | $0 | **≤ 20/09** |
| W6 (27/09) | **E2 measured on Kimi-Linear-48B** (pilot → production, 1×H200 bf16) | ~$250 | **≤ 27/09** |
| W7 (04/10) | Verifier v0 from measured tolerances · PR iteration · interviews | ~$50 | — |
| W8 (16/10) | H100 authoring block only if PR needs it · offers · **audit: the 6 checks below** | ~$150 | **16/10** |
| W9–W16 | **K3 rebuild season** (GDN → KDA → mini → K3-ify → serve) | warchest | — |

**Hardware law:** rent, never buy. 5090 $0.34/hr (iterate, FP4) · H100 $2.50–3.90 (author/profile,
ncu-capable vendors only) · H200 $3.99 (E2, 141 GB = one-GPU home for 48B bf16). **In-window cap
$1,500; warchest ≥$3,500 for interviews/relocation. Every rental names its measurement + consumer.**

## MEASUREMENT — what a stranger can verify on Oct 16

① E1 + E2 public (JSON + ledger, predictions vs measured) · ② PR open (URL) · ③ measured-review
comment live (URL) · ④ ≥24 applications, ≥1 interview loop · ⑤ ≥$1k income (Mercor/PI) ·
⑥ every claim in the CV traces to a number in the repo.

## RULES — everything that survived the simplification

1. **Evidence before claims.** Probe git/URLs; never trust memory or prior sessions.
2. **Predictions before measurements.** Ink first, then run. A result without a mechanism sentence
   doesn't count.
3. **The seal** (agents are locked out, hook-enforced): `reference.py`'s body · kernel bodies under
   active study · RL loss math · verifier tolerances. Everything else — agents execute freely.
4. **One public change per day, pushed before 21:00.** The day scores Y/N on that alone.
5. **Applications fire the moment E1 is public.** No readiness debates.
6. **This file only shrinks or updates in place.** New idea → one line in the backlog, or it dies.
   No new plan documents, no new trackers, no new boards.
7. **Ordering — settled 26/08, do not re-litigate.** From-scratch happens at the *layer under study*:
   the oracle tonight (my hand) · `chunked_wy` rebuilt from blank W3 (my hand) · full K3 rebuild =
   Season 2. Full-stack rebuild-FIRST was already run (Jul–Aug: ~87 planning files, 0 kernels) and is
   the named failure mode. Measurement is not what follows understanding — it is how understanding is
   tested into existence; the only falsifiable mastery metric is prediction error shrinking.

## TODAY (Wed 26/08)

```
rm .git/index.lock
git add -A && git commit -m "refactor: collapse planning corpus to PLAN.md; archive stale generations; harness + instrument lint"
git push origin main            # ← streak dies here
```
Then: Row 000 on paper → 3 predictions inked → the 5 lines → `--self-test` green → `--run` →
ledger close → push = **E1 PUBLIC** → evening: Anthropic Perf-RL + Together SG applications.
Wall: 21:00.

## BACKLOG (one-liners; undated; no ceremony)

- MI300X ROCm day if W7 has slack (~$50)
- B200 day only if a reviewer or interviewer demands a Blackwell number (≤$150)
- AI-Infra hackathon: automated gate-check fires 04/09; ignore otherwise
- Batch-2 applications (NVIDIA new-grad, DeepMind, Fireworks, Baseten…) when E002 is public
- Vietnamese share-doc (`~/Desktop/ke_hoach_60_ngay_first_principles.html`) — regenerate only after E1
- Kernel-path curriculum map: `docs/KERNEL_MASTERY_SPEC.md` (Vizuara workshop fully dissected 26/08,
  live syllabus → our receipts/slots; skip verdict ×4; their week-order = free K3-season quarry skeleton)
