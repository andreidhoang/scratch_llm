# Kernel Mastery Spec — the complete Vizuara-workshop map (2026-08-26)

> **What this is.** Operator-ordered engineering spec: every unit of Vizuara's Kernel Engineering
> Workshop (syllabus fetched LIVE 2026-08-26 16:26 +07 from kernelworkshop.vizuara.ai), mapped to
> this project's receipts, schedule slots, book chapters, and production deliverables — so that
> "master the kernel path, not missing any component" is *verifiable*, not felt.
> **This is a SPEC, not a schedule**: it changes no dates; `PLAN.md` rule 7 governs ordering.
> Sources: live fetch 26/08 · `performance/KERNEL_ROADMAP_2026.md` (session-level dissection,
> 07-14, verdict V2) · the 51-posting demand scan (08-14, ERRATA-D) · curriculum artifact 08-18.

## §1 · The buy/skip verdict, re-tested today (4th test, fresh data)

**SKIP — confirmed again, now with worse economics.** What changed since the last test: the launch
price **expired Aug 15; the price is now $3,300** (+10%). What didn't change: cohort **Oct 12 – Dec 7
= exactly our W9–W16 (the K3 season)**; the syllabus center of mass (GEMM → 93.7% cuBLAS) still
appears in **0/45** job-requirement bullets while its single profiling session covers the market's
#1 ask (38/45); and the live syllabus contains **zero structural components our system lacks**
(map below — three genuine *minor* gaps, all deliberately post-gate). Their "Anthropic partner"
badge is a marketing partnership, not a hiring channel. Economics: $3,300 ≈ **330 hours of 5090
iteration or ~40 B200 hours** — silicon beats seats. The three tests any future buy-impulse must
pass: (1) which PLAN.md check does it advance? (2) does $3,300 beat the same money as metal?
(3) does an Oct-12 start beat an Oct-16 audit? Today: none pass.

## §2 · The complete map — every workshop unit → our system

Legend: **✅** receipt exists in this repo (public, measured) · **🎯** scheduled slot in PLAN.md ·
**🕳️** genuine gap, named honestly · **🔒** deliberately post-gate (K3 season / fence) ·
Market = mentions per 45 postings with full bullets (08-14 scan).

### Part I — Parallelism from the CPU up (their sessions 1–7)

| Their unit | Competency | Market | Ours |
|---|---|---|---|
| CPU parallelism (SIMD/AVX, threads, OoO) | CPU mental model | ~0 standalone (folded into "C++" 30/45) | Book Vol I·1–2. **🕳️ minor: no hands-on SIMD lab** — optional quarry item, zero JD pull; not worth trunk hours |
| L1 Roofline: "how fast can this go" + predict-then-measure lab | The two-number speed limit | **38/45** (as profiling/root-cause) | **✅ deepest asset we own**: ridge computed from own rentals; decode 16→77%-of-roofline story; predict-then-measure is our DAILY law (T6), not a lab. Vol I·4 · brick ⭐#5 · **Row 000 D1 = this, tonight** |
| L2 CUDA model + first kernels + GPU Puzzles + read SASS | Thread/warp/divergence model | 22/45 (CUDA C++) | ✅ repo kernel ladder + warp-sim instrument (1w 9cy/44% vs 4w 18cy/89%); SASS reading = 5090 dev-box reps (W2+); Vol I·2, V·16 |
| L3 Memory hierarchy + transpose ladder + first ncu | Coalescing, bank conflicts, occupancy | in 38/45 umbrella | ✅ coalescing instrument (4/8/32 sectors measured); reduce/transpose rungs in `kernels/`; ncu = W8–10 block on Crusoe/Hyperstack (their own partner vendor). Vol I·3 |

### Part II — GEMM worklog naive → cuBLAS (sessions 8–13)

| Their unit | Competency | Market | Ours |
|---|---|---|---|
| L4–L5 GEMM ladder 1.3%→36.5%→93.7% (coalesce, tile, warptile, float4) | The classic ladder | **0/45 as requirement** | **✅ already climbed**: our ladder 4.1→38.9→**81.9% of cuBLAS on sm_120**, per-rung ledger; their last-12% tier = our W8–10 H100 rungs. Vol V·16 · bricks #17–24, ⭐ mma story. **Re-own via quarry rebuilds, not re-purchase** |
| L6 Tensor cores + WMMA GEMM | mma/WMMA, precision menu | named in Anthropic Perf-GPU JD | ✅ mma.sync rungs in ladder + Vol V·17 (mma.sync→wgmma→tcgen05); fragment-layout instrument (PTX §9.7.14) |

### Part III — Attention & profiling (sessions 14–17)

| Their unit | Competency | Market | Ours |
|---|---|---|---|
| L7 Profiling like a pro + **debug 3 sabotaged kernels** | Nsight/NCU, bottleneck triage | **38/45 — the #1 ask** | ✅ profiling spine (launch-tax 955/token found via profile; cudagraph 15.38→3.96 ms) · **sabotage drill = poached into W5+W7 interview prep** (their best idea, credited since July) · Vol IX·31 |
| L8 Attention + build FlashAttention-1 live | online softmax, tiling attention | FA named in JDs | ✅ **FA2 rebuild owned** (50% of SDPA, 44× leaner mem @8K — stronger than their FA1 target); online-softmax instrument + ⭐ brick; Vol IV·12–13 |

### Part IV — The modern frontier (6 deep-dives, sessions 18–23)

| Their unit | Competency | Market | Ours |
|---|---|---|---|
| DD1 FA from scratch → FA2/FA3 | attention lineage | high | ✅ FA2 owned; FA3 async = W8–10 H100 day; Vol IV·13 |
| DD2 Beating cuBLAS on H100 (TMA, WGMMA, warp-spec) | Hopper last-20% | the "profile & ship on H100s" JD line | **🎯 = our W8–10 csrc ladder verbatim** (cuBLAS→+TMA→+warp-spec→+pingpong, per-rung ledger); C3 arc |
| DD3 Triton → CUTLASS → CuTe-DSL | modern authoring layers | Triton 11/45 · CUTLASS/CuTe 7/45 | ✅ Triton GEMM receipt (134.3% of proxy) + Vol V·18 · **🕳️ minor: CUTLASS/CuTe hands-on = 🔒 K3-season rung by design** (pre-gate it blocks nothing; roadmap §1 holds the 2026 DSL landscape incl. TileLang/FlashQLA facts they don't teach) |
| DD4 Inference-serving kernels (prefill/decode, PagedAttention, spec-decode, FP8/4-bit) | serving stack | 13/45 server internals | **✅ unusually strong**: paged-KV allocator built (3,528 tok/s agg) · continuous batching 2.30× · spec-decode 1.21–1.39× · cudagraph decode · Vol VIII·27–30. **And our E1→PR lands INSIDE vLLM itself — not a lab about it** |
| DD5 Blackwell & NVFP4 (tcgen05, TMEM) | newest silicon | 17/45 low-precision | ✅ NVFP4 vs MXFP4 **measured on own sm_120 rentals** (20.43 vs 18.74 dB — consumer Blackwell has FP4; H100 doesn't) · Vol V·19 · tcgen05/TMEM = 🔒 B200 trigger (≤$150, armed) |
| DD6 FlashAttention-4 (adaptive softmax, petaflop) | the newest kernel | prestige | ✅ C5 arc + our fact-check they don't teach (FA3-FP8 hit ~1.2 PF in 2024; FA4's 1.6 PF is the bf16-Blackwell landmark) · 🔒 bench post-gate |

### Part V — AI-written kernels (sessions 24–25)

| Their unit | Competency | Market | Ours |
|---|---|---|---|
| DD7 DeepSeek FlashMLA & DeepGEMM | small-sharp-kernel school | culture | ✅ Vol V/VIII citations + K3-season CuTe/DeepGEMM rung; MLA weight-absorption already in `mla.py` |
| DD8 LLM-generated kernels: KernelBench, AlphaEvolve, agent+profiler loop | **supervising AI kernel-writers** | the 2026 frontier | **✅ we hold the sharper version**: KernelBench-Verified facts (374×-ReLU fraud; corrected best 0.88×), MLSys-2026 agent-assisted>full-agent, and **our verifier capstone (tolerances from measurements) IS the missing piece their module describes**; C7 arc · M-AI box in roadmap §M-AI |

### Part VI — Capstone (with Crusoe, graded demo day)

| Theirs | Ours |
|---|---|
| A Crusoe-suggested problem, solved in-cohort, **graded by instructors at demo day** | **E1 divergence map → measured review on live vLLM PR #45819 → own PR → E2 on Kimi-Linear-48B → verifier — graded by vLLM maintainers, CI, and hiring loops.** Same partner-company hardware (Crusoe is already our ncu vendor, $3.90/hr) — rented directly, without the $3,300 wrapper |

**Gap audit result: 0 structural gaps.** Three minor ones, all deliberate: hands-on SIMD lab
(≈0 market pull → optional quarry), CUTLASS/CuTe hands-on (🔒 K3 season, blocks nothing pre-gate),
FA4 internals bench (🔒 post-gate trigger). Everything else exists here as a *measured receipt or a
dated slot* — where their version is an exercise, ours is a public artifact with an external consumer.

## §3 · What we take from them (poach list, updated with today's fetch)

1. **Sabotage-kernel drill** — in the plan since July (W5, W7). Their best idea.
2. **Predict-then-measure as the roofline lab** — is our daily law (T6/Row 001), deeper than a lab.
3. **"Why it matters / where it's used / who uses it" card format** — the Measured Stack already
   carries it per chapter; keep enforcing on every new write-up.
4. **Their week-order as the K3-season quarry default** (Part I→V maps cleanly onto W9–W16 mornings)
   — a free syllabus skeleton for Season 2, worth exactly $0.

## §4 · Standing decision

The workshop is a **map of territory we either already hold with receipts or have deliberately
fenced with dates and triggers.** Buying the map for $3,300 while standing on the territory is not
mastery — shipping tonight's oracle is. Re-open this verdict only if all three §1 tests flip.
