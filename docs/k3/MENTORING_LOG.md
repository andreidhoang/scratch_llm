# K3×KERNELS MENTORING LOG — cross-session ledger

> Ledger, not plan (sanctioned by HERMES rules). One row per mentoring session. The mentor agent
> reads this at session start (cross-session memory of the merged program), updates it at session
> end. Program of record: `docs/k3/MERGED_KERNELS_K3_ROADMAP.md` · style binding: root `CLAUDE.md`
> amendment 2026-08-10. **Rule: a checkpoint the Navigator fails or skips returns within 48h
> (spaced repetition); a PASSED checkpoint is recorded with date.**

## Current position
- **Target locked (2026-08-11):** Anthropic RE, Performance RL ($350–850k, greenhouse 5160330008).
  Every node scored by "does this make that application undeniable?" Strategy of record: `FAST_TRACK_2026.md §6`.
- **APPLICATIONS GATED (Rev 2.2, operator decision 2026-08-11, mentor dissent on record):** no
  employment application until the six-item Mastery Gate is green (FAST_TRACK §3 Lane 3; scope-locked;
  hard review Day 90 = 2026-11-08). Weekly recorded defense rep from W1 counts toward G5. Money lanes
  (Mercor ≤10 hr/wk, PI bounties) continue — not applications.
- **The spine (all work collapses to this line):** Layer 0 roofline → understand KDA state →
  K2 `core/kda.py` → **KDA decode kernel ≥85% mem roofline (the ONE flagship public number)** →
  wrap in KernelGym (RL env grading kernel-writing) → apply. Flagship ladder rows: K39 → K37 → K18.
- **Active merge node:** M1→M2 — KDA × memory-bound kernels; the KDA decode kernel is now the explicit target.
- **Next lesson:** Lesson 3 — open `linear_attn.py` (three-path contract) and begin the K2 `core/kda.py`
  rebuild, once the KDA-state crossover hand-derivation (assigned Lesson 2) is returned and reviewed.
- **Blocking build:** `core/kda.py` (hand-built; K2 critical path) · `csrc/fundamentals/reduction_warp.cu` (K26).
- **Operating rule:** first action of the day cannot fail (10 min, binary); one spine node per deep block;
  ONE recorded defense rep per week from W1 (counts to gate G5) — the daily mock machine stays retired;
  Mercor ≤10 hr/wk. One process metric: days something public changed.
- **Ledger row:** `KERNEL_MASTERY_2026.md` §4 row 2 — S1 prediction PRE-REGISTERED 2026-08-10
  (naive ≤35% HBM; fused 85–90% HBM). Awaiting measured.

## Lesson ledger

| # | Date | Lesson | Core content | Checkpoints set | Status |
|---|---|---|---|---|---|
| 0 | 2026-08-10 | The memory pyramid + roofline intro | 4 tiers, human-time scale, ridge = peak/bw, minimal byte bill, fusion > FLOP-opt for memory-bound | pyramid cold · roofline sketch w/ 3 kernels placed · coalescing spoken answer · S1 pre-registration | PRE-REGISTERED (row 2); answer key delivered; Navigator self-grading PENDING |
| 0-R | 2026-08-10 | FULL RESET from silicon | two verbs (move/combine) · memory wall physics · latency vs bandwidth (truck fleet) · SIMT/warp · pallet rule (128B line) · toy 4-lane machine (arrangement A vs B) · reduction tree · roofline re-derived | 5 cold questions (truck analogy; 32×4B=128B; L2-can't-save-A; vector-add AI; tree cost) | PENDING Navigator answers |
| 1 | 2026-08-10 | Coalescing & GEMV ladder | thread-per-row (stride-N, 11.7% DRAM, L1 96% busy≠useful) → block-per-row (54.9%) → two-stage reduction (85–90%) → float4 (97.2% = cuBLAS); occupancy-not-the-goal; one-big-lever law | draw both access patterns · warpReduceSum tree · ncu paradox 1-liner · 90→97% k-forensics answer · build `reduction_warp.cu` | PENDING |
| 2 | 2026-08-11 | Link the whole picture: roofline → K3 → production → the job | (1) the 7-layer foundation tree; (2) roofline re-derived on the Navigator's OWN card (ridge = 72.1/0.551 = 131 FLOP/B) with GEMM/attention/decode placed; (3) K3 read as three roofline moves — KDA/MLA/MoE/MXFP4 = (a)+(c); (4) one-token tensor trace through mini-K3 (shapes + move + number/row); (5) the spine collapses KernelGym = KDA-decode-kernel = Performance-RL-JD into one object; (6) principal review of the K3 plan (co-design = the staff inflection; the trap = 12wk/25hr not-started; corrected stale premises across 4 docs) | **decode roofline hand-derivation** (params×bytes ÷ 0.551 TB/s = tok/s ceiling; % of 253) · **KDA-state crossover** (full-attn `2·H·d·2·t` vs KDA const `16·64·64·2`; find crossover t) · one hostile sentence: "what did KDA give up vs full attention, which K3 layers buy it back?" | PENDING Navigator answers |

## Method notes (what works for this Navigator)
- Reset-style full-from-zero rebuilds (Lesson 0-R) landed better than compressed sessions — default to
  toy-scale (4-lane) machines before real-scale numbers.
- Every number must be traceable: book page, `bench/RESULTS.md` row, or a `python -c` run. No asserted
  numbers.
- **Bilingual Feynman is BINDING (2026-08-11, operator request "once and for all").** Every main
  English paragraph and every diagram/visual gets a Vietnamese complement directly beneath it,
  Feynman-style: simplest possible words, one concrete everyday analogy, from first principles, no
  unexplained jargon, and name the "aha" so it sticks the first time. English = the defense/interview
  layer (terms, shapes, numbers); Vietnamese = the intuition layer. Test: if the VN can't explain it to
  a smart 12-year-old, the concept isn't mastered yet — that gap is the lesson. Never drop the VN to
  save space; interleave, don't append at the end.
