# Operating Rhythm — the frontier-engineer day, run on this harness

> **What this is.** The *dynamic* layer on top of the static harness. `CONTEXT_ENGINEERING.md`
> explains why every `.claude/` file exists (the instruments); `IMPLEMENTATION_PLAN.md` says *what*
> to build (the score); **this doc is the conductor** — the daily/weekly cadence that fuses how top
> frontier optimization engineers/researchers actually spend a day with the commands you already have.
> Lever 2 (on-demand). Read once; thereafter `/standup` and `/eod` enact it.

---

## 0. The operating model — how a frontier optimization engineer actually works

Distilled from the FOP (`CLAUDE.md`), research-as-MDP (`CONTEXT_ENGINEERING.md §3.9`), and
`FRONTIER_PRACTICE_2026.md`. Eight invariants — the *day* is built to satisfy them, not a todo list.

1. **One active node.** The day orbits the single highest-variance *open* node — one experiment, one
   load-bearing step — chosen by EV (success-rate ÷ time), not a queue. Pull the signal node; defer
   the deterministic scaffolding. (Right now: ZeRO-1 / A2-D2, *or* SFT masked-CE / A5 — never both in one day.)
2. **Predict-before-run is the open of every run.** Write the falsifiable number/shape/bound *first* —
   the roofline bound (comm vs flop vs memory) and the number, before launch. It is simultaneously the
   debugging anchor and the learning anchor. A run with no pre-written prediction is wasted signal.
3. **Measured > implemented.** A result exists only when *measured/profiled*. For perf work the DoD is
   a **profile**, not a green test. A day ends with a measured artifact **or** a cleanly falsified
   hypothesis — never just "I wrote code."
4. **Overlap human attention with machine time.** Frontier work is gated by long jobs. Launch the
   run/profile, then do CPU-side work while it runs — *never idle-wait*. The GPU burst (§3) is the
   sharpest form of this.
5. **Execution > analysis; watch the doc-to-code ratio.** Ship beats plan. No new doc without a
   same-day commit hash. When doc/code climbs, stop writing and resolve a node. The standing failure
   mode is **the treadmill** (scaffolding instead of building) — `/next` and this rhythm exist to break it.
6. **Pre-committed kill criteria.** Every speculative path states its abandon threshold *before* you
   start. A failed node is abandoned, not sunk-cost. "What would falsify this?" is the cheapest
   state-estimate that collapses MDP uncertainty fastest.
7. **Citation-tree traversal, not cover-to-cover.** When a paper matters, triage in minutes: the
   claim · the method · the one figure. Reuse before re-deriving; don't rebuild owned work.
8. **Mastery rides the build (teach-back gate).** Don't advance a concept until you can teach it back
   — a *concept* gate, never a commit gate. Green-CI is the only commit gate. Mode-3 targets (RL-math,
   the kernel you're learning) you write yourself; the agent gives failing tests / critique / review.

---

## 1. The daily loop — Open → Deep Work → Close

```
  ┌─────────────┐     ┌──────────────────────────────┐     ┌─────────────┐
  │  OPEN ~10m  │ ──► │     DEEP WORK (the loop)      │ ──► │  CLOSE ~10m │
  │  /standup   │     │  /master · predict · test ·   │     │   /eod      │
  │ pick 1 node │     │  build · /profile · /ship     │     │ durable +   │
  │ + predict   │     │  (overlap GPU time, §3)       │     │ next node + │
  │ + kill-crit │     │                               │     │ /clear      │
  └─────────────┘     └──────────────────────────────┘     └─────────────┘
```

### OPEN (`/standup`, ~10 min) — commit to ONE node
- Orient: branch + uncommitted + what's green (`STATUS.md`).
- **Choose the single EV-ranked node** for today (one experiment / one load-bearing step).
- **Write the falsifiable prediction** (the number/shape/bound) and the **kill criterion** for it.
- Output is a one-paragraph contract for the day. Then `/next` builds it.

### DEEP WORK (the core loop, one node) — `§2`
The `CLAUDE.md` "How we build" loop, run on the one node, with the run-discipline folded in.

### CLOSE (`/eod`, ~10 min) — make state durable, then reset
- Update `STATUS.md` (what is green *now*); commit it if the day produced code.
- Write any *non-obvious, hard-won* fact to project memory (not what the repo already encodes).
- **Name tomorrow's single node** (so `/standup` starts warm).
- `/clear` — external memory > chat memory; reset to a clean, high-signal window for tomorrow.

---

## 2. The core deep-work loop (one node, test-first)

```
 1. /master <concept>      first-principles + 3 lenses (shapes · system · worked example) + teach-back
                           └─ Mode-3 (RL-math / kernel)? YOU write it; agent → failing test / critique only
 2. PREDICT-BEFORE-RUN     write the number/shape/bound first  (the day's prediction, refined per step)
 3. TEST-FIRST             encode the invariant as a test (loss-at-init · overfit-one-batch · causal-
                           no-leak · 2-rank gloo equivalence · P[minhash]≈Jaccard · RL-logging present)
 4. BUILD → GREEN          implement; ruff + pyright + pytest -m "not gpu"
 5. MEASURE (perf nodes)   /profile → roofline-analyst → BOUND / WHY / NEXT / PREDICT  (DoD = a profile)
 6. /ship                  ship-reviewer (correctness + scope) → green-CI → it hands YOU the commit
 7. TEACH-BACK GATE        explain it back + modify-and-predict one variation; only then advance
```

Anti-idle rule: if step 5/anything launches a long job, immediately pick up the next CPU-side sub-task
(another test, the next module's first-principles, a `STATUS.md` edit) — never watch the bar fill.

---

## 3. GPU development discipline (standing GPU; measure continuously)

**We develop on a standing GPU** (RTX PRO 4000 Blackwell, sm120, 25 GB). No step waits for hardware —
the measure-fix-remeasure loop runs every day. Keep two habits that still pay even with the GPU present:

- **Correctness before the number:** make the kernel/step green against its oracle (often `-m "not gpu"`
  on the box — pure-torch oracle, gloo equivalence, fake-quant) *before* you trust a measurement. A green
  test is the floor; a profile near the roofline you predicted is the result.
- **Predict → run → capture → diagnose:** write the falsifiable number first; run under
  `cuda.synchronize()` + warm-ups; capture raw artifacts (ncu/nsys, throughput CSVs) to `profile/`;
  then `/profile` → roofline-analyst for BOUND/WHY/NEXT and log predicted-vs-measured to `bench/RESULTS.md`.

Rails for *this* card: **25 GB cap** (size models/seqs to fit) and **report "% of *this* Blackwell," never
imply datacenter numbers**. **Rent a bigger / multi-GPU box (`vastai`) only for what this card can't do**
— full-scale throughput vs H100/B200, real multi-GPU NCCL, the DELTA datacenter-roofline (Step-0). The
NVFP4-on-state numerics seam, by contrast, is *best run here* (native Blackwell FP4).

---

## 4. Weekly cadence

| Day | Ritual | Output |
|---|---|---|
| **Mon — pick the spike** | State the week's EV bet (one A-pillar node *or* a 🔵 build-lab) + its P1–P7 predictions + kill criteria. | A pre-registered week-contract. |
| **Tue–Thu — deep work** | The daily loop on that node. CPU-build toward green. | Green commits; a teach-back-passed concept/day. |
| **any day — measure** (the GPU is standing) | §3 — predict → run → capture → `/profile`. | Measured numbers in `bench/RESULTS.md` + `STATUS.md`. |
| **Fri — postmortem + prune** | What was *falsified*? Doc-to-code ratio this week? Subtract-before-add: delete one stale doc/test/path. Refresh `STATUS.md`. | A clean board; next week's candidate nodes. |

---

## 5. The four metrics that keep the rhythm honest

Glance at these at `/eod` and on Friday — they catch drift before it compounds.

1. **Active-node count = 1.** More than one experiment in flight → you've left the MDP discipline; cut to one.
2. **Doc-to-code ratio (this week).** New doc lines without same-day commit hashes → treadmill; stop writing, resolve a node.
3. **Predicted-before-run rate.** Fraction of runs with a pre-written number. Target 100%; a run without one taught you less than it cost.
4. **Measured-vs-implemented.** Count perf/RL claims that are *measured* vs merely *implemented*. Only the measured ones are results (`[FACT]`); the rest are `[INFERENCE]` until profiled.

---

## 6. Mapping to the current state (2026-06-29)

Two live nodes; the rhythm says **pick one per day**, EV-ranked.

- **A5 SFT masked-CE** (highest-EV ship). `/standup` → node = "`response_mask` + `sft_microbatch_train_step`
  correct." Prediction: loss-at-init on the SFT head ≈ log V over unmasked positions; overfit-one-batch → ~0.
  Mode-3 — *you* write the masked-CE body; I supply the failing tests (`response_mask` alignment,
  masked-mean, overfit). Wire `utils/monitors.py` logging *before* any GRPO run.
- **A2-D2 ZeRO-1** (systems finish, = DELTA Phase-1 infra). Node = "optimizer-state sharded across ranks,
  2-rank gloo equivalence to single-process AdamW." Prediction: per-rank optimizer memory ≈ (1/N)× the
  unsharded state; param/grad unchanged. Test-first: the 2-rank gloo equivalence (×5 seeds). Pair it with
  the **100B memory one-pager** (params+grads+Adam ≈ 16–20 B/param → why DP+TP+PP).

A3/A4 are thin slices only — not day-nodes until A5 ships and the A2 distributed half is green.

---

## 7. Anti-patterns (name them so you catch yourself)

- **The treadmill** — building harness/docs instead of CS336 code. Antidote: `/next`, the doc-to-code metric.
- **Idle-wait** — watching a run/profile finish. Antidote: §2 anti-idle rule, §3 overlap.
- **Green ≠ done (for perf)** — shipping a kernel on a passing test with no profile. Antidote: step 5, DoD = profile.
- **Unpredicted run** — launching before writing the number. Antidote: `/standup` + step 2; it's a hard gate.
- **Sunk-cost node** — nursing a failing path past its threshold. Antidote: the pre-committed kill criterion.
- **Two experiments at once** — split attention, no clean signal. Antidote: active-node = 1.
- **Mode-3 outsourcing** — letting the agent write the RL loss / the kernel you're learning. Antidote: FOP-6; agent refuses, gives a failing test instead.
