# Kernel Mastery Spec — v2 (2026-08-26 evening)

> **What this is.** The ONE kernel-path document: the complete Vizuara-workshop map, the
> late-Aug-2026 live market scan, the verified 2026 production-stack fact ledger, the
> consolidated curriculum spine (the untracked root draft `KERNEL_MASTERY_ROADMAP.md` is
> harvested here and archived — PLAN.md rule 6), and the mentoring program that delivers it.
> **This is a SPEC, not a schedule**: it changes no dates; `PLAN.md` governs ordering and wins
> every conflict. Vision chain + wall-map: CLAUDE.md North Star (top of file) · artifact
> https://claude.ai/code/artifact/e3fc28db-aacd-4c84-99eb-39f6c131c756
> Sources: v1 (commit 64c11e8, live syllabus fetch 26/08 16:26) · 5-agent research pass
> 26/08 evening (site ×5 fetches, N=21 live JDs, stack primary sources, GitHub landing-zone
> status, in-repo receipts audit) · `performance/KERNEL_ROADMAP_2026.md` (07-14) ·
> 51-posting scan (08-14). Confidence labels per FOP-4: [FACT]/[INFERENCE]/[UNCERTAIN].

## §1 · The buy/skip verdict — 5th test (fresh instructor/logistics data)

**SKIP — reconfirmed, now on stronger grounds.** Everything from tests 1–4 held under a
5×-refetch dissection (price $3,300 current, launch price expired Aug 15 [FACT]; cohort
Oct 12 – Dec 7 = exactly W9–W16; GEMM-to-cuBLAS still ≈0 as a market requirement — see §2;
zero structural gaps — see §5). **New facts, all pushing the same way:**

- **Instructors [FACT]:** Raj Dandekar (MIT PhD, ~10 concept lectures) + Shubham Panchal
  (~15 coding sessions) — Panchal's stated credentials are **mobile/on-device ML** (Android
  since 2017, SmolChat, Rust/C++/Kotlin); the page lists **no production GPU-kernel
  engineering experience for either instructor**. The competency the workshop sells is not
  evidenced in its own faculty page.
- **Schedule [FACT]:** Mon/Wed/Fri 7:00–9:00 AM, **timezone unstated anywhere**. IST ⇒
  8:30–10:30 ICT (eats the K-path morning deep block); PT ⇒ 21:00–23:00 ICT (eats the
  ship-before-21:00 wall + evening lane). Either reading collides with the two-block day
  three days a week for nine weeks.
- **"Anthropic partner" [FACT]:** logo-level company claim, scope unstated, not a workshop
  endorsement. Crusoe's concrete role: capstone project ideas + one guest session. Cloud
  GPU vendor for the cohort: **not named on the page**; no credit amount promised.
- **Zero third-party coverage [FACT]:** searches for independent reviews/announcements of
  this specific workshop return nothing — too new to have a track record.

Economics unchanged: $3,300 ≈ 330 hrs of 5090 iteration or ~40 B200 hours; silicon beats
seats. The three tests any future buy-impulse must pass: (1) which PLAN.md check does it
advance? (2) does $3,300 beat the same money as metal? (3) does an Oct-12 start beat an
Oct-16 audit? Today: none pass, and test 3 now also fails on timezone arithmetic.
Re-open only if all three flip; nothing in tonight's data flips any.

**6th test — full-page re-fetch 29/08 [FACT], nothing flips.** Confirms every prior fact
($3,000 launch expired Aug 15 → **$3,300 current**; Oct 12 – Dec 7; Mon/Wed/Fri 07:00–09:00,
**timezone still unstated**; 25 × 2 h; Crusoe = capstone ideas + one guest session). Three
new facts, all pushing SKIP harder: **(a) prerequisites are "Python + basic PyTorch, no CUDA
or C++ assumed"** — the workshop's floor is below our A3 receipt (mma.sync GEMM at 81.9% of
cuBLAS), so weeks 1–4 are re-teaching owned ground; **(b) "no GPU required — all live builds
run on cloud GPUs"**, i.e. a headline inclusion whose marginal value to us is **$0**, since
the sm_120 card is the daily driver and rentals are already budgeted per §6; **(c) it is
"chapter 3 of a GPU trilogy"** (after 5D Parallelism · Inference Engineering) — the two prior
chapters cover ground this repo holds with receipts (`bench/RESULTS.md` distributed + serving
rows). Non-price inclusions worth naming honestly: the Vizuara Kernel Engineering Book +
an interview-prep site + recordings — content assets, none of which produce a public artifact
a maintainer can review. **The skip verdict now rests on 6 independent passes.**

## §2 · The 2026 market truth (live scan, N=21 postings with full bullets, 26/08)

Prior baseline = 51-posting scan (08-14). N=21 is deliberately kernel-targeted, so trust
**rank changes and vocabulary**, not raw rates. [FACT] unless noted.

| Competency | 08-26 live | 08-14 scan | Read |
|---|---|---|---|
| Profiling / root-cause (Nsight, roofline) | **17/21 (81%)** | 38/45 #1 | **Still #1** — and escalated to interview *format*: Modal literally asks for a "tell us a story about boosting GPU performance" war story |
| CUDA C++ (memory hierarchy, warp reasoning) | 15/21 (71%) | 22/45 | The floor for every dedicated kernel req (OpenAI, xAI, NVIDIA, TML, Anthropic-GPU) |
| Serving-stack internals (vLLM/SGLang/TRT-LLM/**FlashInfer**) | 10/21 (48%) | 13/45 | Rising; "deep dive into underlying codebases" (Baseten); "contributions to vLLM" = NVIDIA new-grad standout; FlashInfer named for the first time |
| Distributed + NCCL/NVLink/IB | 10/21 (48%) | — | Anthropic-GPU, OpenAI, Fireworks, Meta |
| Triton | 8/21 (38%) | 11/45 | Always alongside CUDA/CUTLASS, never alone |
| Low-precision FP8/FP4/INT8 | 7/21 (33%) | 17/45 | Flat; "custom kernels for emerging quantization formats" (Anthropic) |
| torch/JAX internals, compile, custom ops | 7/21 (33%) | — | Anthropic-GPU, Meta-PyTorch, xAI (pybind→JAX/XLA) |
| CUTLASS/CuTe family — **now "CUDA/TileIR/CuTeDSL/cutlass/Triton"** | 6/21 (29%) | 7/45 | **Up + renamed**: TileIR/cuTile are new JD vocabulary this cycle (NVIDIA new-grad reqs); tile-level DSLs are the abstraction NVIDIA hires into |
| ML compilers (XLA/MLIR/TVM/JIT) | 6/21 (29%) | — | Groq, NVIDIA kernel-libraries ("JIT domain specific compilers") |
| Attention kernels / FlashAttention | 4/21 (19%) | — | Anthropic "co-designing attention mechanisms for next-generation hardware" |
| Accelerator-agnostic / multi-accelerator | 4/21 (19%) | new | Anthropic Staff+ Inference Runtime $405–485k owns a GPU/TPU/Trainium runtime; "porting workloads between accelerators" desired on Perf-RL |
| **Correctness / numerics / determinism — explicit** | **3/21 (14%)** | **0 — NEW AXIS** | Anthropic Inference Systems: "*genuine interest in correctness as an engineering discipline: numerics*"; xAI: "ensuring correctness while considering floating point errors". **The batch-invariance era reached JD language.** |
| RL × inference convergence | 2/21 (10%) | new | Anthropic **Performance-RL** (RL envs to teach models accelerator coding, $350–850k); Together Turbo ("inference-aware training") |
| AI-assisted kernel generation | 2/21 (10%) | new | NVIDIA "kernel code generators"; Anthropic Perf-RL is the supervising-the-writer role |
| GEMM-from-scratch, explicit | 1/21 (5%) | 0/45 | xAI only, verbatim ("from scratch or by utilizing CuTe/CUTLASS"); still nobody asks "% of cuBLAS" |
| Linear-attention / SSM / hybrid kernels | **0/21** | 0/45 | **Labs ship them (Qwen, Kimi) but JDs can't ask for them — a differentiator you bring, not a requirement you meet** |
| MoE kernels, explicit | 0/21 | — | Implicit only (expert parallelism inside distributed phrasing) |

Comp calibration [FACT]: kernel-titled IC roles cluster $280–850k (Anthropic), $350–475k
(TML evergreen), $310–460k (OpenAI), $180–440k (xAI), $248–292k (Groq); NVIDIA new-grad
$124–241.5k. Market churn: Together's GPU-programming req became "Core ML (Turbo)";
Cerebras kernel req 404s; Moonshot kernel hiring is Beijing-side; Groq (an LPU company)
now hires CUDA/ROCm GPU-inference engineers.

**Strategic read:** the program's exact bet — *measured numerics of a shipping recurrence,
landed in a serving stack* — sits at the intersection of the #1 competency (root-cause),
the new correctness/numerics axis, the RL×inference theme (batch-invariant GDN demand is
RL-determinism demand, vLLM #48613 says so verbatim), and the linear-attention white space
no JD knows how to ask for. Nothing in the fresh scan argues for reordering; everything
argues the E-lane artifacts are aimed at the exact 2026 hiring surface.

## §3 · The 2026 production stack — verified fact ledger

All [FACT] against primary sources fetched 26/08 unless labeled.

**Authoring layers.** CUTLASS **4.8.0** (released 25/08!; 4.7 added experimental CuTe-DSL
*Primitives API* + warp-spec task scheduling + compile-time spill/sync diagnostics; 4.8
adds initial Rubin SM107 FP8/FP4). **CuTe-DSL (Python)** is in public beta graduating
~end-of-summer, isomorphic to CuTe C++, ~20–30× faster compiles; FA4 is written entirely
in it — the de-facto frontier authoring layer, while C++ CUTLASS remains the substrate of
shipped libraries [INFERENCE on "de-facto"]. **Triton 3.7.1** (3.8 due ~today, 3.9
Dec 15); **Proton** in-tree profiler (CGO 2026 paper) — intra-kernel scoped timing that
runs on sm_120; **Gluon** in-tree lower dialect (explicit layouts/smem/pipelining) with
official tutorials and AMD ROCm adoption — the ladder *inside one toolchain* is now
Triton → Gluon → PTX. **Helion 1.4** (PyTorch "higher-level Triton", autotunes schedules).
**ThunderKittens 2.0** (Blackwell + NVFP4; Ampere dropped); **HipKittens** = the AMD port.

**Attention lineage.** FA3 = the stable Hopper production kernel (C++/CUTLASS; one of
SGLang's three deterministic backends). **FA4** [FACT]: released ~2026-03-05
(arXiv:2603.05451), **entirely CuTe-DSL Python**, up to 1605 TFLOP/s bf16 fwd on B200 =
71% util; softmax pipelining w/ dedicated correction warpgroup, exp split across
MUFU.EX2 + FMA polynomial, conditional rescaling; cuDNN 9.13+ absorbed its techniques.
**FA4 does not run on sm_120** (wheel crashes at JIT [INFERENCE]) — every FA4 hands-on is
B200-rental-gated. **FlashInfer** = the JIT attention-engine layer shared by vLLM/SGLang
(paged-KV + radix unified block-sparse format) — teach "attention backend" as an
interface, not a kernel.

**Blackwell ISA lineage — the #1 port-breaking interview fact.** `mma.sync` (Ampere+) →
`wgmma` (**sm_90a ONLY** — datacenter Blackwell drops it) → `tcgen05` + **TMEM** (256
KB/SM accumulator memory, single-thread-issued async MMA, **sm_100 ONLY**). **sm_120**
(RTX 5090 / RTX PRO): NO wgmma, NO tcgen05/TMEM; HAS `mma.sync` incl. **block-scaled
NVFP4/MXFP4/FP8 variants** + single-CTA TMA (`cp.async.bulk.tensor`); smem 99 KiB/block
vs 228 on sm_100, cluster size 1 [secondary source on the numerics]. Hopper kernels do
NOT port by recompiling. **NVFP4** = E2M1 + FP8-E4M3 scale per 16-block + FP32 tensor
scale; MXFP4 = E2M1 + E8M0 per 32-block — format bit-layout is now an ISA-level object
(our own receipt: NVFP4 beats MXFP4 by 1.48×, `bench/RESULTS.md:545`).

**Small-sharp-kernel school.** DeepGEMM: fully-JIT (per-shape first-call compile), Apr-2026
release added Mega MoE kernels + FP8×FP4 GEMM + PDL. FlashMLA: publishes its roofline
position with the kernel (3000 GB/s memory-bound / 660 TFLOP/s compute-bound, H800) —
exactly this repo's "DoD is a profile" standard, in the wild.

**Batch-invariance is a shipping feature class** [FACT]: TML "Defeating Nondeterminism"
(root cause = batch-size-dependent reduction strategies) → `batch_invariant_ops` →
SGLang deterministic inference (~34% overhead, 3 backends) → `VLLM_BATCH_INVARIANT=1`
(motivated verbatim by "reproducible/true-on-policy RL"). The uncovered remainder is
linear-attention layers — the program's landing zone (§4).

**Linear/hybrid attention in production** [FACT]: Qwen3-Next-80B introduced GDN at 3:1
linear:full, now the workhorse across Qwen3.5/3.6; **Kimi Linear** (arXiv:2510.26692)
ships **KDA** (per-channel gating) 3:1 with MLA, 48B/3B-active, −75% KV, up to 6× 1M-ctx
decode, day-0 vLLM support. **fla** library: KDA since v0.4.0 (Oct 25), current v0.5.2
(Jul 26, KDA cached-inference); `fla/ops/kda/` = `naive.py` + `fused_recurrent.py` +
`chunk*.py` + `wy_fast.py` — **three implementations of K2's recurrence to diff against,
plus the exact WY territory of E001**. GDN-2 + fused AttnRes landed May 2026. Counter-
weight: **MiniMax M2 reverted to full attention**, publishing why — recurrent states are
precision-sensitive under low-precision serving, infra integration immature. That
published reversion is the strongest external justification of numerics-first sequencing.

**AI-written kernels — real vs hype** [FACT]: KernelBench-Verified (arXiv:2607.16241):
best frontier-model geomean falls 1.43× → **0.88×** under a hardened protocol — no model
consistently beats PyTorch against honest baselines; 28% of "wins" raised peak memory.
Sakana's 10–100× claim was publicly walked back (eval-sandbox exploit; 1.13×→0.82× on
re-eval). What's real: AlphaEvolve's verified 32.5% FA speedup via search + hard verifier
+ profiler loop. The scarce skill is **building the verifier and knowing where numerics
break** — the exact artifact this program's capstone produces.

**Shipping a kernel into torch, 2026 checklist** [FACT]: `@torch.library.custom_op` →
`register_fake` (compile/export tracing) → `register_autograd` → `torch.library.opcheck`
(contract, NOT numerics) → `assert_close`/`gradcheck` → benchmark under `torch.compile`.
Triton kernels ride `torch.compile` natively without the wrapper [INFERENCE — verify
current tracing constraints before that lesson]. Every kernel rung ends with this list —
"kernel done" includes integration.

## §4 · Corrections to our own record (claims honesty — found by tonight's pass)

1. **The "13%" anchor is a misreading.** FLA issue #389 (filed **2025**-05-06, now
   CLOSED, no visible root-cause fix) reports `o diff: 0.130267, ratio: 0.006313` —
   **0.130 ABSOLUTE max diff = ~0.63% RELATIVE**, chunked-Triton vs recurrent, bf16,
   B=4 T=128 H=1 D=16, at the pathological gate log α = 0. It resurrects #104; the
   gate-at-0 boundary is a twice-reported, never-root-caused soft spot. → **PLAN.md's WHY
   line "disagree (13% on record)" needs a one-line fix by Huy's hand.** Calibration
   consequence: the literature anchor for chunked-vs-recurrent divergence is ~0.6%
   relative in bf16 at the worst gate — Row-001 predictions should be argued against
   *that* scale, on *our* shapes (T=512, d=64), not against 13%.
2. **The landing zone is no longer empty — it is crowded and unresolved** (CLAUDE.md's
   "zero linked PRs" note is stale). Current state [FACT, 26/08]: issue **#42960** OPEN,
   unassigned, A100 repro/test offer still live; **PR #45819** (broad: per-sequence
   loops + `use_cp=False` pin) OPEN under active review, **no approvals**, rebased 08-24 —
   review thread found bs≈60–62 divergence traced to FLA/Triton reduction-order
   sensitivity, a contributor named **three independent divergence sources** and proposed
   cross-chunk state carry + fixed reduction widths, and reviewers demanded the official
   determinism suite; **PR #49827** (Qwen-specific backend: 64-token chunk alignment,
   packed recurrent decode, +6.74% throughput, needs-rebase) competes; issue **#48613**
   frames the same gap as bitwise-deterministic RL post-training; **#50680** shows the
   failure surface extends into cudagraph padding paths. **Strategy consequence: the
   entry is not a third parallel PR — it is the measured map that adjudicates between
   #45819's reduction-order hypothesis and #49827's chunk-alignment hypothesis, delivered
   as the review both threads lack.** Precedent that this works: vLLM #41292 (KDA state-
   layout regression shipped through five releases because nothing diffed chunked-prefill
   state against a reference — reporter's diff-vs-reference methodology is our template).
3. **Receipt-legend violations in spec v1** (its own legend says ✅ = in-repo, public,
   measured): the warp-scheduler instrument (9cy/44% vs 18cy/89%), the coalescing
   4/8/32-sector instrument, and the PTX-fragment-layout instrument exist **nowhere in
   the repo** — they were claude.ai mentoring artifacts. Re-marked ⬜ (out-of-repo,
   unreceipted) in §5. The online-softmax stepper (`performance/viz/`) is a
   *visualization* receipt; the measured receipt is the FA2 rows. All other v1 receipts
   verified with file:line: GEMM ladder 4.1→38.9→81.9% (`bench/RESULTS.md:917-919`),
   FA2 50%/44× (`:892`), Triton GEMM 134.3% **of cuBLAS-proxy** (`:516`), NVFP4 (`:545`),
   paged-KV 3,528 tok/s (`:146`), continuous batching 2.30× (`:89`), spec-decode
   1.21–1.39× (`:384-385`), cudagraph −74.3% 15.38→3.96 ms (`:420`) + ~955 kernel
   launches/token eager (`:27`) — two configs, don't merge them in one breath.
4. **Same-day doc tension, named:** `MERGED_KERNELS_K3_ROADMAP.md`'s ERRATA-G (26/08)
   declares the fused KDA decode kernel **not novel** (five shipped implementations:
   Moonshot FlashKDA, Qwen FlashQLA, FlashInfer `fused_kda_decode`, vLLM fused kernel,
   SGLang `cutedsl_kda`) and M0–M6 inert-until-W9, while PLAN.md (same day, and by its
   own line 3 it wins) re-promotes "KDA decode kernel ≥85% memory roofline" as flagship.
   Resolution on paper: PLAN.md wins; the flagship is a **mastery artifact measured
   against five production references, not a novelty claim** — write it up that way.

## §5 · The complete Vizuara map — every unit → our system (updated 26/08 evening)

Legend: **✅** receipt in this repo (public, measured, file:line verified tonight) ·
**🎯** slot exists in PLAN.md's dependency order · **⬜** out-of-repo artifact or
unreceipted (was wrongly ✅ in v1) · **🕳️** genuine gap, named honestly · **🔒**
deliberately fenced (post-gate / rental trigger). Market = §2 live scan.

### Part I — Parallelism from the CPU up (sessions 1–7)

| Their unit | Market | Ours |
|---|---|---|
| S1 CPU parallelism (SIMD/AVX, threads, OoO) | ~0 standalone | Book Vol I·1–2. 🕳️ minor: no hands-on SIMD lab — zero JD pull, optional quarry |
| L1 Roofline + predict-then-measure lab | **#1 (81%)** | ✅ deepest asset: ridge from own rentals; decode 16→77% story; predict-then-measure is daily law (T6). **Lesson 1 delivered 26/08 on E001 Row 000 — their L1 instantiated on a production recurrence** |
| L2 CUDA model + first kernels + SASS | 71% | ✅ repo kernel ladder; ⬜ warp-sim instrument (claude.ai artifact, not in repo); SASS reps = 5090 dev-box; Vol I·2, V·16 |
| L3 Memory hierarchy + transpose + first ncu | in #1 | ✅ reduce/transpose rungs in `kernels/`; ⬜ coalescing-sector instrument (out-of-repo); ncu = Proton-on-sm_120 now also viable (§3); Vol I·3 |

### Part II — GEMM naive → cuBLAS (sessions 8–13)

| Their unit | Market | Ours |
|---|---|---|
| L4–L5 GEMM ladder 1.3%→36.5%→93.7% | 1/21 explicit (xAI) | ✅ climbed: 4.1→38.9→**81.9% of cuBLAS sm_120** (`RESULTS.md:917-919`). Their 93.7% > our 81.9%: **explicit raise-the-bar target ≥90% for the re-own rebuild**, not a silent contradiction. Last-12% tier = H100 rungs |
| L6 Tensor cores + WMMA | Anthropic-GPU JD | ✅ mma.sync rungs + Vol V·17; ⬜ fragment-layout instrument (out-of-repo). New JD vocab to speak: TileIR/cuTile/CuTeDSL (§2) |

### Part III — Attention & profiling (sessions 14–17)

| Their unit | Market | Ours |
|---|---|---|
| L7 Profiling + **debug 3 sabotaged kernels** | **#1 (81%)** — and now an interview *format* (Modal war-story) | ✅ profiling spine (launch-tax→cudagraph receipts) · sabotage drill = poached (their best idea) · add **Proton scoped-timing lab** (new, runs locally) |
| L8 Attention + FA1 live | 19% | ✅ FA2 owned (50% SDPA, 44× mem) — stronger than their FA1 target; online-softmax: math receipted via FA2 rows, stepper viz in `performance/viz/` |

### Part IV — Modern frontier (deep-dives 1–6)

| Their unit | Market | Ours |
|---|---|---|
| DD1 FA→FA2/FA3 | high | ✅ FA2 owned; FA3 async = H100 rental day; Vol IV·13 |
| DD2 Beating cuBLAS on H100 (TMA/WGMMA/warp-spec) | the "profile & ship on H100s" line | 🎯 H100 csrc ladder (cuBLAS→+TMA→+warp-spec→+pingpong), per-rung ledger; **teach wgmma as sm_90a-ONLY — dropped on Blackwell (§3)** |
| DD3 Triton→CUTLASS→CuTe-DSL | 29%, vocabulary rising | ✅ Triton GEMM receipt (134.3% of proxy) · **update: the 2026 ladder is Helion→Triton→Gluon→CuTe-DSL; add the same-GEMM-in-Triton-then-Gluon rung (new, local)** · CUTLASS/CuTe hands-on 🔒 K3-season by design |
| DD4 Inference-serving kernels | 48% | ✅ unusually strong: paged-KV 3,528 tok/s · cont-batch 2.30× · spec-decode 1.21–1.39× · cudagraph −74.3% · **and our lane lands INSIDE vLLM (§4.2) — not a lab about it**. FlashInfer = the layer to name (§3) |
| DD5 Blackwell & NVFP4 (tcgen05/TMEM) | 33% low-precision | ✅ NVFP4>MXFP4 measured on own sm_120 (1.48×) · tcgen05/TMEM 🔒 B200 trigger (≤$150, armed) · **sm_120 capability matrix now [FACT]-verified (§3)** |
| DD6 FA4 | prestige | ✅ C5 arc + corrected lineage facts; **FA4 = CuTe-DSL Python, 1605 TF/s = 71% B200 util, does NOT run on sm_120** → read the source locally, numbers 🔒 B200 trigger |

### Part V — AI-written kernels (deep-dives 7–8)

| Their unit | Market | Ours |
|---|---|---|
| DD7 DeepSeek FlashMLA & DeepGEMM | culture | ✅ Vol V/VIII + MLA absorption in `mla.py`; DeepGEMM = the JIT-specialization case study; FlashMLA = publish-the-roofline-with-the-kernel exemplar |
| DD8 LLM-generated kernels | 10% explicit, rising | ✅ we hold the sharper version: KernelBench-Verified **0.88×** + Sakana walk-back + AlphaEvolve-what's-real (§3), and **our verifier capstone (tolerances from measured envelopes) IS the missing piece their module describes**. Anthropic Perf-RL is this as a $350–850k job |

### Part VI — Capstone

| Theirs | Ours |
|---|---|
| Crusoe-suggested problem, graded at demo day | **E1 divergence map → the adjudicating review on #45819 vs #49827 (§4.2) → own follow-up fix or regression harness → E2 on Kimi-Linear-48B (fla v0.5.2 `ops/kda` as the three-way diff target) → verifier `tolerances.yaml` from measured envelopes** — graded by vLLM maintainers, CI, and hiring loops. Crusoe is already our ncu vendor at $3.90/hr, rented directly |

**Gap audit: 0 structural gaps** (unchanged through 6 tests). Minor + fenced: SIMD lab
(🕳️ optional), CUTLASS/CuTe hands-on (🔒 K3-season), FA4/tcgen05 bench (🔒 B200 trigger).
Newly added local rungs from the 26/08 stack pass: Proton lab, Triton→Gluon GEMM,
torch.library shipping checklist (§6).

**Unit-numbering caveat (29/08 re-fetch):** the live page is internally inconsistent about
session counts (Part I is described as "4 sessions" while its own week table lists sessions
1–3; Parts II–III likewise slip by one). **Map at part/unit level, never at session number**
— no session index in this spec carries a [FACT] label. What is stable and confirmed: 6
parts, 8 numbered lectures each paired with a coding session, 8 deep-dives, 1 graded capstone.

**Their per-unit card format, adopted verbatim as our lesson-close template** (poached §8):
*why it matters · where it is used · who uses it*. Ours adds the fourth field their cards
lack — **the receipt** (`file:line` or a `bench/RESULTS.md` row). A card without a receipt is
a brochure; that fourth field is the entire difference between their map and our territory.

## §6 · The consolidated curriculum spine (root draft harvested; PLAN.md orders it)

The archived draft's ten load-bearing contributions, kept as **undated** spec material
attached to PLAN.md's dependency order (K2→K6 → M0→M6; E-lane backlog):

**Operating principles (bind every kernel rep, effective immediately):**
1. Profile before optimizing (nsys timeline → ncu kernel → now also Proton scopes).
2. **Lock clocks before benchmarking** (`nvidia-smi --lock-gpu-clocks`) — unlocked clocks
   make every A/B a lie.
3. Correctness ladder before speed: fp64 reference + measured rel-err/ULP · gradient test
   if backward exists · adversarial set (NaN/Inf, seq_len=1, odd shapes, non-contiguous).
4. `torch.compile` is the floor a custom kernel must beat to earn maintenance cost.
5. One variable per run, written prediction before every run (= T6, already law).
6. Every kernel gets a WORKLOG version table (version · % of roofline · the one change ·
   the ncu/Proton counter explaining it) — in `bench/RESULTS.md`, not a new file.
7. Time-boxed kill criterion per optimization idea; log negative results.
8. Rental discipline: prepare locally → exact experiment list → rent → capture
   `.ncu-rep`+CSV → kill instance. GPU-hours are for measurement, not thinking.
9. **Ship-into-torch checklist closes every kernel** (§3): custom_op → register_fake →
   register_autograd → opcheck → assert_close/gradcheck → compile benchmark.

**Hardware plan (§3 facts; prices/ncu re-verified live 27/08):** 5090/sm_120 = daily
driver ($0.33 Vast KVM for ncu; mma.sync, block-scaled FP4/FP8, single-CTA TMA,
Triton/Gluon/CuTe-DSL, Proton) · H100/sm_90a = wgmma/TMA/warp-spec rental block
($2.50–3.90, Hyperstack/Crusoe ncu-capable) · B200/sm_100 = tcgen05/TMEM/FA4/DeepGEMM day
(**$4.6–6.1/hr; Verda $6.11 VM = ncu path; full day ≈ $40–60**; fires after the review
lands) · GB300 1× rentable (Verda $8.62) if a reviewer demands the freshest silicon ·
ncu-blocked everywhere containers rule: RunPod pods, Modal, Vast default docker. Split
every rung into "runs on sm_120" vs "rental-gated" at design time.

**Arch-routing rule (30/08 — PLAN.md § Hardware law is the law; this is the working form).** Split
by the **kernel's target arch**, never by the biggest card. A larger card is not a superset:
`-arch=sm_120` will not load or JIT on sm_90 (PTX compat is forward-only), and a Triton kernel on
another arch recompiles to different SASS + autotune configs — a different instance, whose counters
discharge nothing about the original. **Measured consequence:** 4 of the 5 `ncu`-debt metrics
registered in `bench/` on 04/07 were filed to "the H100 day" and sat **57 days**; they are sm_120 and
discharge on the $0.33/hr daily driver above. Only the WGMMA tensor-pipe metric is genuinely Hopper.
This makes **gap 2 (no public ncu profile — the #1 hiring axis, 10/10 JDs) a ~$1 one-session item**,
not an E2/N5 dependency. Corollary: same compute capability ≠ same card — the ledger's rows are 70
SMs / 0.551 TB/s, a 5090 is ~170 SMs / ~1.8 TB/s; re-run R0 to re-anchor peaks before comparing.

**Harvested rungs (each = one future lesson + one measured artifact; slot into PLAN.md
phases, never into dates):**
- **GEMV / batched-GEMV decode twin** (draft §2.3; absent from Vizuara AND spec v1):
  ≥90% of memory bandwidth + the tokens/sec-ceiling derivation. Natural neighbor of the
  KDA decode kernel — same wall. → attaches to M1/M2 reps.
- **Epilogue fusion** (bias+GELU in-kernel) — the CUTLASS concept JDs name. → M2.
- **FA backward** (logsumexp trick, recompute, atomics-vs-split) — training-role demand;
  Vizuara barely touches it. → M3.
- **MoE kernels**: grouped GEMM, token permute/unpermute, fused expert epilogues
  (DeepGEMM Mega-MoE as reference). → M4/S29.
- **Integration layer**: the §3 torch.library checklist + one vLLM/SGLang forward-pass
  code-read (Python → kernel launch, where a custom kernel plugs in). → with E3.
- **Triton→Gluon rung** (new tonight): same GEMM, compiler-decides → you-decide;
  layouts/pipelining made explicit. Runs on sm_120. → M2.
- **Proton lab** (new tonight): scoped intra-kernel timing on an existing receipt kernel;
  reconcile against ncu. → M1.
- **Agent+profiler loop** (draft phase 5, hardened by §3 facts): Claude writes → compile
  + correctness gate (verified-protocol style: honest baselines, memory accounting) →
  ncu/Proton metrics fed back → failure analysis published. Trains the supervising-the-
  writer role Anthropic Perf-RL hires for. Requires the verifier first — capstone order
  already guarantees that.
- **DELTA capstone spec** (draft phase 6) — preserved as *extension material* under the
  PLAN.md flagship: hypothesis ≥1.5× at decode B≤32 vs composed baseline, four named
  baselines, evals-locked-first, kill at <1.15×. Subordinate note per §4.4: fused
  KDA/GDN decode kernels have five shipped implementations — any DELTA-style fusion work
  is measured against those references, framed as mastery-with-receipts, not novelty.
- **Raise-the-bar:** GEMM re-own target ≥90% cuBLAS (vs 81.9% receipt) when the quarry
  rebuild fires.

## §7 · The mentoring program (how every lesson runs)

Modality is fixed by the standing contract (CLAUDE.md T-loop + M1–M4 scoring; PRR;
Vietnamese Feynman blocks; visualization built from real numbers; L2 menus with hidden
ranking; prediction precedes every measurement): **frame → build-from-zero (code-anchored
`file · func · line`) → worked NEIGHBOR example (never the target) → technique + trap →
hand back the target → predict → run → reconcile.** Depth lives in the chat lessons and
`docs/k3/MENTORING_LOG.md`, not in this spec. Sealed remains sealed: oracle, kernel bodies
under study, harness, RL loss math, verifier tolerances — Huy's hand.

Lesson ladder = PLAN.md's own order (each lesson closes on that phase's gate):
- **L1 delivered 26/08 · gate pending — Roofline + the recurrence + the oracle** (E001
  Row 000/001; = Vizuara L1 on live stakes). Gate: Row 000 filled on paper, predictions
  inked, 5 lines, self-test, teach-back. ✅ only when the gate passes.
- **L2 — K2a: per-channel decay.** Scalar α → `Diag(α)`; scaled-sigmoid g_min=−5 vs
  negative-softplus; the A_log **[128]-vs-[num_heads]** checkpoint trap (FACTS A18 — the
  #1 silent bug). Gate: fp64 reference in `tests/test_k3_kda.py` with per-channel α.
- **L3 — K2b: the input path.** Short causal conv k=4 ring buffer (off-by-one = silent
  receptive-field shift) + Swish + L2Norm (what makes the eraser a projection). Gate:
  conv ring-buffer decode ≡ prefill.
- **L4 — K2c: chunkwise WY/UT at chunk 64.** The E001 instrument generalized to
  Diag(α); fla `ops/kda/{naive,fused_recurrent,chunk}` as the three-way diff. Gate:
  chunkwise ≡ recurrent ≡ fp64-ref; then fla parity fwd+bwd (GPU day).
- **L5–L8 — K3–K6** per PLAN.md rows (gated-MLA absorption identity · AttnRes
  online-softmax merge · SiTU/LatentMoE bounds · mini-K3 loss-at-init/overfit-one).
- **M-series lessons** attach the §6 harvested rungs at their M0–M6 homes; E-lane lessons
  (divergence map → #45819/#49827 adjudication review → E2 → verifier) fire when Huy
  re-prioritizes the backlog (PLAN.md's call, not this spec's).

## §8 · Poach list & standing decision

Poached (all cost $0): sabotage-kernel drill (their best idea — now also self-inflicted
per draft §3.1: break your own K8 kernel, diagnose from ncu alone next morning) ·
predict-then-measure-as-lab (our daily law, deeper) · why-it-matters card format ·
their Part I→V week-order as the K3-season quarry skeleton · demo-day framing ("closest
thing to an on-site before the on-site") — that is what the vLLM review thread is, with
real maintainers · **GPU Puzzles** (29/08 — their coding-session-2 lab is Sasha Rush's
free open-source puzzle set: `srush/GPU-Puzzles`, browser/Numba, ~14 puzzles ending at a
tiled matmul; $0, no cohort needed, and it is the cheapest way to pressure-test the CUDA
index-arithmetic that the K2 chunk kernel will demand — slot it as a warm-up before M1,
not as a rung of its own).

**Standing decision, 5th confirmation:** the workshop is a map of territory we either
hold with verified receipts (§5) or have deliberately fenced with dates and triggers.
Its faculty page cannot evidence the competency it sells; its schedule collides with the
two-block day in every timezone reading; its $3,300 buys ~330 5090-hours of measurement.
Buying the map while standing on the territory is not mastery — tonight's oracle is.
Re-open only if all three §1 tests flip.

## §9 · The production operating system (researched 26/08 evening — 3-agent pass, primary sources; binds every perf task under the production-first redirect)

### 9.1 The daily loop as practiced [FACT — vLLM/SGLang/PyTorch/TML/Anthropic sources]

**Monitor-first.** (0) The day starts from a dashboard delta, not a profiler: vLLM runs a
nightly perf+accuracy matrix (separate `vllm-project/perf-eval` repo — 17 model-hardware
recipes on H200/B200/MI300X/MI355X, TTFT/TPOT via vllm-bench + GSM8K/GPQA/AIME + BFCL)
surfaced at perf.vllm.ai/ci.vllm.ai, with a CI-analyzer bot that diffs nights, walks
intervening commits, and posts culprit + auto-revert PR (~70% accurate); PyTorch's nightly
Inductor runs land in ClickHouse on hud.pytorch.org with Grafana alerts. (1) **Profile**:
timeline (nsys) → kernel table; SGLang's discipline: benchmark each kernel in a unit test
as its *theoretical upper bound* and optimize the gap to in-system time; isolate phases
structurally (PD-disaggregated, "unlimited resources for the non-tested phase").
(2) **Hypothesize**: name the bound + predict against an **external anchor** (SGLang
tracked "20% slower than DeepSeek's profile" as the standing gap); falsify folk hypotheses
with minimal counterexamples first (TML: `torch.mm` on `a[:1]` vs batched-then-slice
differs by 1669.25 — two lines killed "concurrency+floating-point"). (3) **Fix one
variable** under the §9.2 regimen. (4) **Verify with three simultaneous gates** — vLLM's
release bar: correctness CI + E2E serving perf at fixed QPS (serving numbers carry ~5%
noise; a microbench win that doesn't move TTFT/TPOT is not a result) + accuracy eval
(speed with degraded GSM8K = rejected). Quantify per-optimization by ablation (SGLang: TBO
27–35%, EPLB 1.49×/2.54× — never one blended claim). (5) **Prevent**: pinned environments
(one Docker image, lock files) so tomorrow's diff is attributable; the nightly becomes the
permanent guard. Meta-lesson (InferenceMAX nightly re-benchmarks; Anthropic's three-bug
postmortem, misrouting peaked at 16% of Sonnet traffic): numbers decay in days, pre-deploy
evals miss production degradation — date-stamp + version-stamp every claim, re-run stale
baselines. **Solo binding: every rep runs this loop scaled down — `bench/RESULTS.md` is
the nightly; the ledger row is the dashboard.**

### 9.2 The variance-control regimen [FACT — Standard Kernel, measured; triton do_bench docs]

Persistence mode on · lock clocks **below the throttle point and verify under load** (a
1980 MHz lock silently throttled at the power ceiling; 1200 MHz held) · flush L2 with a
dummy write > cache (>50 MiB on H100) · 100 warmup / 500 timed reps · **median or 10%
trimmed mean, never mean** on heavy-tailed data · isolated GPU (contention: +21% mean,
30× std-dev) · kernels <10 µs cannot be reliably timed → `do_bench_cudagraph` ·
`do_bench` default warmup underestimates non-autotuned jit kernels ~30% (triton #2306) ·
cloud "A100 80GB" rentals were measured as PCIe/SXM4/GRID variants with non-overlapping
matmul runtimes — verify the actual SKU. Standardize on ONE harness (do_bench) and record
driver/CUDA/clock state with every number.

### 9.3 The landed-PR evidence standard [FACT — cross-verified on 6 repos with named PRs]

The universal shape of a MERGED perf/numerics PR: **(1) correctness gate stated and run
FIRST** (bitwise or reference-tolerance, fwd AND bwd where gradients exist; sanitizer
output for memory bugs) · **(2) same-hardware before/after** with exact GPU, shapes, and a
reproducible command · **(3) root-cause narrative**, not symptom description · **(4) scope
honesty** — a limitations section naming what was NOT measured · **(5) a NOT-A-DUPLICATE
statement** naming competing open PRs and differentiating (2026 norm) · **(6) AI-assistance
disclosure** + "submitter reviewed every changed line" (vLLM: `Co-authored-by` trailers on
top of DCO; FlashInfer: reviewers may reject if the author cannot walk the rationale).
**Model PRs to imitate:** vLLM #53247 (external; 19,200-point sweep, 4 pre-registered
gates, determinism-tax table, competing-PR paragraph — merged in 3 days, author added to
the #27433 checklist) · vLLM #45683 (NCCL reduction-tree root cause + alternatives-considered
with bandwidth math) · fla #1135 (one-line KDA mask fix: compute-sanitizer proof,
adversarial test the old suite couldn't see, honest 0.36% cost — merged in 5 h) · fla
#1125 (Intel-gated GDN tune, per-kernel + varlen + E2E tables, identical checksums —
merged in **90 minutes**) · flashinfer #4686 (A/B/A CUPTI metrology on an 8-line diff).
**Stall mode:** vLLM #31548 — mixed benchmarks (0.44× regressions beside 6.3× wins) +
large integration surface = 5 months, closed unmerged. **Evidence completeness predicts
merge latency more than author standing does.** Repo mechanics: vLLM DCO + `ready`
label/`/ci run` (CI never auto-runs) + RFC required >500 LOC + status cadence 2–3 days ·
sglang run-ci label + GSM8K sanity + oncall merge model · fla naive.py-parity +
NaN-poisoned conftest + benchmark-bot comment + [tag] commits · flashinfer `!claim`
self-assign + template-as-squash-commit · FA: no process — scope + parameterized tests +
adjacency are the currency · PyTorch: CLA + mergebot + years-scale (know-it-discuss, not a
fast landing zone).

### 9.4 Who gets hired off this — the pattern [FACT]

Tri Dao (FA → Together Chief Scientist) · Zhuohan Li (vLLM → OpenAI gpt-oss infra → Meta)
· Zihao Ye (FlashInfer → NVIDIA; MLSys'25 best paper; NVIDIA adopted the project) ·
Lianmin Zheng + Ying Sheng (SGLang → xAI inference team) · Horace He (PyTorch compilers →
TML, authored the blog that created the entire batch-invariance workstream) · **Songlin
Yang (fla → TML; KDA co-author with Moonshot — the exact lineage this program studies)** ·
counter-example: Woosuk Kwon converted vLLM standing into a $150M-seed company. **The
pattern: every documented hire was attached to an OWNED SUBSYSTEM with measured numbers —
never scattered PRs.** The trust ladders (vLLM committer bar: ~30 substantial PRs + ~10
reviews of others + material perf improvements; sglang oncall nomination; fla's CI-enforced
header as the maintainer list; flashinfer `!claim`→collaborator) are the institutional
mirror. **This program's subsystem: linear-attention numerics/determinism — one connected
reviewer graph across vLLM/sglang/fla/flashinfer (zhiyuan1i: fla maintainer landing vLLM
hybrid-KV PRs; fla kernels are the pinned-commit baseline in flashinfer #4001 and sglang) —
one well-evidenced landed PR is visible to all four.**

### 9.5 The live surface map (verified 26/08 evening; freshest, least-claimed first)

| Surface | Where | State | Fit |
|---|---|---|---|
| **#48613 — GDN batch-invariant validation harness** | vLLM | OPEN, unassigned. **Correction to our framing: it asks for batch-invariant kernels + a solo-vs-batched validation harness — NOT logprob-divergence (that's the surrounding RL ecosystem: #41733, vllm-omni #4864)** | **The direct production continuation of E001. Claim by commenting a concrete harness design WITHIN DAYS** — a flashinfer-GDN-prefill commit is already in vLLM CI and #49827 is adjacent |
| **#27433 — batch-invariance tracker** | vLLM | OPEN, 76 comments, explicit 🙋Help-needed TODAY: FLASHINFER_MLA invariance (fi #2107) · batch-invariant perf · model-coverage validation (good-first-issue) · AMD #52231 · XPU #49209 · spec-decode determinism #52522 | **The formal on-ramp: land one sub-item → named on the checklist → assigned the next** (LioEinaudi precedent) |
| #51562/#51483 — GDN metadata builder classifies stateless first chunk as decode | vLLM | OPEN, **0 comments** | Chunked-prefill state classification — exactly the instrument's territory |
| #49918 — spec-verify FULL cudagraph skips GDN recurrent-state write → deterministic garbage | vLLM | OPEN | State-handling numerics |
| #31720 — hybrid-GDN+AWQ degenerates at temp 0 on sglang; same checkpoint clean on vLLM | sglang | OPEN | **A cross-ENGINE divergence characterization made for the E001-style instrument** |
| #35150 — accumulated GDN state drift under DSpark forced-reject | sglang | OPEN | Spec-decode × linear-attention numerics |
| Batch-invariant FlashInfer GDN prefill | spans vLLM+sglang+flashinfer | **Unowned** — sglang reverted to Triton because FlashInfer's path is batch-sensitive on SM90 (#35632/#34859) | The cross-repo prize the map aims at |
| fla fp16 chunk-path tolerance failures reproduce on main (from #1135's own test report) · varlen gradient coverage | fla | Unclaimed | Free characterization task in the home repo of the lineage |
| **flashinfer #2577 — NVFP4 `mm_fp4` returns silent all-zeros on SM120, all backends** | flashinfer | OPEN | Silent-wrong-answer correctness bug **reproducible on the local sm_120 card for $0**; a root-caused repro comment alone is a consumed artifact. Adjacent: vLLM #31085/#47749/#33333 (check #49011's prototype before claiming that slice) |
| **`vllm-project/recipes` `moonshotai/Kimi-Linear.md` is BARE** — zero measured numbers, no HW configs, no determinism section | vLLM recipes | Acknowledged gap; renders at recipes.vllm.ai | **E2 ships as this recipes PR**: H200 TTFT/ITL/tok-s/$ + determinism appendix. $16 smoke test (2–4 h H200) decides shape; full sweep $120–200. TP1 boot at 128–256K ctx is [INFERENCE] until the smoke run; if `VLLM_BATCH_INVARIANT=1` refuses to boot on KDA, **that refusal + unguarded divergence numbers ARE the publishable result** |
| GPU MODE KernelBot | Discord | Always-on, free donated compute; 2026 big-money cycle past (AMD $1.1M, NVFP4 hackathon); fall drop plausible | Visibility complement; kernelbot-data is a HF dataset labs train on |
| Paid lanes | — | Mercor "CUDA Engineering Expert" $300/task min 20 h/wk (OSS PRs explicitly preferred = the entry ticket) · tinygrad bounties ($50k+ paid historically) · Prime Intellect = RL *environments* $100–5k, not kernels | P5; merged PRs are the currency that converts |

**Sequencing (research-endorsed, fits PLAN.md order):** ① harness claim on #48613 ($0,
days — right after E1 is public) → ② $16 H200 smoke test same week (decides the recipes-PR
shape) → ③ sm_120 NVFP4 (#2577) as the parallel deep lane → ④ GPU MODE opportunistic.
Budget consumed ~$150–300 of $1,500. Flag from the research, verbatim discipline: do NOT
copy serve flags from search summaries into commands (two hallucinated flags were caught);
every number date-stamped; every claim [FACT]-labeled against a fetched source.

### 9.6 Workspace zones — where the engineering physically happens (set 26/08 evening)

**Zone 1 — `~/Desktop/scratch_llm` (this repo, PUBLIC): everything that is OURS.**
Instruments + drivers + evidence. `src/scratch_llm/mastery/` = the E-series instrument ·
`experiments/eNNN_*.py` = one numbered driver per experiment (e001 exists; e002 = the
vLLM-facing GDN/KDA invariance harness prototype) · `results/` = public JSON evidence,
committed · `src/scratch_llm/k3/core/kda.py` + `tests/` = the K2 reference (Huy's hand) ·
`bench/RESULTS.md` = the ledger · `performance/rental/` = rental-day runbooks + kernels
(E2's H200 runbook lands here beside the H100/B200/8×H200 ones) · `deploy/` = pod
lifecycle (`00_setup_vast.sh → 01_launch.sh → provision.sh → sync_*.sh`).

**Zone 2 — `~/Desktop/oss/<repo>`: upstream forks — the production landing zones.**
Created 26/08 (empty until a lane opens). Clones of OUR forks of `vllm`, `recipes`,
`flash-linear-attention`, `flashinfer` — cloned when the first PR/harness port begins
(first: `oss/vllm` + `oss/recipes`, after E1 is public). Work there follows THEIR
conventions (§9.3: DCO sign-off, their test layout, their benchmark format), never ours.
**Upstream code is never nested inside scratch_llm**; prototypes graduate from Zone 1
`experiments/` into Zone 2 as proper upstream tests/PRs.

**Zone 3 — the rented pod: ephemeral, measurement-only.** The loop: build + rehearse in
Zone 1/2 → push to GitHub (repo is public — pods clone with no keys) → pod bootstraps
(`scripts/bootstrap-pod.sh`, `docs/VASTAI_BOOTSTRAP.md`) → run the pre-written experiment
list from the runbook → artifacts (`.json`, `.ncu-rep`, logs) rsync back into Zone 1
`results/` + one `bench/RESULTS.md` row → **destroy the pod**. Nothing is authored on a
pod that isn't committed back through git the same session; GPU-hours are for measurement,
not thinking (§6 rental discipline).

## §10 · The mastery operating system — how frontier engineers actually master, and the one method that binds every turn (26/08 late evening)

> **Relationship to standing law, stated first so this is consolidation, not a sixth
> method:** the **T-loop order is untouchable** (settled 08-14, "do not re-litigate": frame
> → build-from-zero → worked NEIGHBOR → technique+trap → hand back → predict→run→
> reconcile). **M1–M4 are operator-owned and unchanged.** This section adds the two layers
> the standing law implied but never named — the *practitioner moves* (what a frontier
> engineer's head actually does) and the *altitude ladder* (the zoom structure of every
> explanation) — and wires the five existing doors (`/op`·`/master`·`/feynman`·`/tutor`·
> `/kviz`) into one method. Attribution discipline: [FACT] where a primary source is in
> hand; [INFERENCE] pending the 26/08 verification pass.

### 10.1 The practitioner moves — reverse-engineered, one per situation

How the people whose artifacts we study actually build understanding. **The move is
triggered by the situation, not scheduled:**

All attributions below verified against primary sources 26/08 (2-agent pass; corrections in §10.8).

| Situation | The move | Practiced by [all FACT unless noted] |
|---|---|---|
| Meeting a new mechanism | **Build the minimal atomic version from scratch, spelled out; no copy-paste, reference allowed** — "when you actually build something from scratch, you're forced to come to terms with what you don't actually understand and you don't know that you don't understand it" | Karpathy (micrograd→nanoGPT→nanochat; Dwarkesh 10/2025, verbatim). Feynman's blackboard at his death (Caltech Archives image 1.10-29): "What I cannot create, I do not understand" — AND its usually-omitted second line: **"Know how to solve every problem that has been solved"** |
| Choosing what/how to learn | **Project-first, depth-wise, on-demand — never bottom-up breadth-first; teach/summarize everything in your own words**; effort is the signal ("Learning is not supposed to be fun… seek the meal — textbooks, docs, papers") | Karpathy (tweet 2020-11-07 verbatim; "shortification of learning" post 2024-02-10) |
| Any performance question | **Napkin-math the bytes and FLOPs BEFORE profiling**; classify compute-/memory-/overhead-bound — "knowing what regime you're in allows you to narrow in on optimizations that matter" | Horace He, "Making Deep Learning Go Brrrr From First Principles" (2022) |
| Designing an algorithm | **Derive it FROM the hardware constraint** — cite the paper, not folklore: "a missing principle is making attention algorithms IO-aware — accounting for reads and writes between levels of GPU memory" (arXiv:2205.14135 abstract); A3 forces A1 | Tri Dao (FlashAttention; same reasoning documented for Mamba's SRAM-resident state) |
| A plausible folk theory | **Kill or confirm it with a minimal counterexample** — two lines of `torch.mm` ended "concurrency+floating-point" | Horace He / TML |
| A confusing phenomenon | **Shrink it to the smallest toy that still shows it** — E001 on a CPU IS this move | toy-models culture; this repo's instrument |
| Reading a paper / abstraction | **Rederive it yourself, from an explicit inventory of ignorance** ("NOTEBOOK OF THINGS I DON'T KNOW ABOUT", weeks "disassembling each branch… looking for the raw edges" — Gleick); **track every abstraction with a concrete evolving example** ("I keep making up examples… my hairy green ball thing" — SYJ 1985); perturb: what breaks if this term vanishes | Feynman (both first-person/biographer documented); `/master` step 4 citation-traversal |
| Entering a codebase | **Trace ONE forward pass end-to-end**; unit-test each kernel as its theoretical upper bound | SGLang discipline (§9.1) |
| A single load-bearing kernel | **The iterative worklog against the vendor library**: naive → rung-by-rung, measured GFLOPs each rung ("optimizing SGEMM iteratively is one of the best ways to deeply understand the hardware") · 2025 vintage: fix the speed-of-light number first, correct-naive second, then let ncu name each next bottleneck | Simon Boehm (siboehm.com worklog, 93.7% cuBLAS); Thien Tran/gau-nernst (FA-on-5090 worklog) |
| Checking understanding | **Write the long-form explanation yourself as the act of learning** ("I'm documenting my learning notes in this blog since 2017" — Weng, her words; distillation as first-class research — Olah, "Research Debt") | Weng; Olah; the worklog culture |
| Stuck | **Shannon's 1952 toolkit, verbatim: cut to essentials · find solved P′ · restate in many forms · generalize every answer · break the jump into small jumps · invert (work backwards)** · Polya's 4 phases · Hamming (Bellcore 1986): work on important problems you have an attack on; compound daily thought | Shannon; Polya; Hamming |
| Meeting a master's artifact | **The reverse-engineering loop (§10.3)** | this program, on FA4/DeepGEMM/fla |

### 10.2 The Altitude Ladder — the zoom structure of every explanation

Five fixed altitudes. **Teach top-down; the frontier derives bottom-up** — that asymmetry
is the single sharpest fact in this section: a senior engineer *explains* A0→A4, but the
designs they're famous for ran A4→A1 (the silicon forced the algorithm — FlashAttention,
tcgen05-shaped FA4, chunk-64 WY). Teaching that only goes down is a tour; mastery is being
able to run the ladder in both directions.

| | Altitude | The question it answers | Binds to |
|---|---|---|---|
| **A0** | Vision / production | What breaks in the real world without this? Who consumes it? | T1 frame · M1 · PLAN.md's consumer rule |
| **A1** | Mechanism | What constraint FORCES this design? Which alternatives lose, and why? | T2 build-from-zero · VN Feynman (b) |
| **A2** | The real code | `file · func · line` at HEAD, as written, gaps named | M4 · code-anchored law |
| **A3** | The machine | Where does every byte live? Layout, movement, roofline position | §9.2 regimen · the byte-counting method |
| **A4** | The number | One runnable/measured value that pins it — never asserted | T6 · FOP-4 measured>implied |

A concept counts as *presented* only when all five altitudes are touched or a deferral is
named ("A4 lands on the GPU day"). Altitudes are **content structure, not interaction
order** — the T-loop still governs who speaks when.

### 10.3 The reverse-engineering loop — for artifacts by masters (FA4, DeepGEMM, fla)

① Read only the **interface + claims** (README, function signatures, the headline number)
→ ② **predict the internals on paper** — what would *you* build under their constraints;
where must the hard part be → ③ read the real thing → ④ **diff your prediction** — every
divergence is either a move you didn't know or a constraint you didn't model; the diff IS
the lesson (fluency reading skips exactly this) → ⑤ **extract the move** into §10.1's
table, named, with its trigger. This section's own table is the loop's output applied to
the 26/08 research corpus.

### 10.4 The Feynman gate — for HARD concepts (provenance-honest)

The popular "4-step Feynman Technique" is posthumous packaging [FACT — Scott H. Young,
~2011, never written or named by Feynman]. What Feynman *documented*: rederive everything
yourself from an explicit inventory of ignorance (Gleick), track every abstraction with a
concrete evolving example (SYJ 1985), and the blackboard creed with both lines. Our
operationalization, fired whenever a red-flag phrase appears ("it just works", "I'd have
to look it up", a hand-wave under probing):

1. **Plain words, in Vietnamese** — restate to a smart 12-year-old; the VN layer is the
   intuition layer and the test (MENTORING_LOG method note: if the VN can't carry it, the
   concept isn't owned yet — that gap IS the lesson).
2. **Why THIS structure and not the alternatives** — the constraint, not the description.
3. **Gap-hunt** — find the step in your own explanation that hand-waves; that step is the
   next question, never skipped.
4. **Shrink + falsify** — the smallest toy that still shows it, plus the counterexample
   that would kill your story if it's wrong.

### 10.5 The mastery bar — cited, not invented

`MASTERY_LEDGER.md:4-6` already defines it: **TRACED ≠ DEFENDED** — "rebuild it from
blank, predict its number, defend every choice against 'why not X', and say what breaks at
10×." §10 adds exactly two clauses: **explain it at all five altitudes (VN included)**,
and **locate it in the production frontier** (who ships it, what they trade — §9's
surface). The only falsifiable mastery metric remains prediction error shrinking (PLAN.md
rule 7).

### 10.6 One method, five doors — when each command fires

`/op` **T-loop** = the shape of EVERY turn (default; never opt-in) · `/master <concept>`
= deliberate deep-dive on one concept (T-loop + full ladder + citation traversal +
teach-back gate) · `/kviz` = when a mechanism deserves an interactive stepper and no
maintained tool covers it (use-first check; predict-before-advance) · `/feynman` = the
**inverted** door — the human talks ≥80%, Claude plays the confused student, 5-dimension
grade, MASTERY_DEBT ledger; the exit gate of `/kviz` and the periodic re-check of any ✅ ·
`/tutor` = the Socratic fallback when stuck (narrows the question, never hands the
answer). Escalation: stuck in `/feynman` → `/tutor` → re-enqueue at +48 h; a rescued
explanation is not a cleared one.

### 10.7 The per-turn contract (the ≤10 lines that bind every explanation — mirrored in CLAUDE.md)

1. Open at **A0**: the production frame — consumer, what breaks (T1/M1; never a task list).
2. Derive **A1** from the forcing constraint; name the losing alternatives.
3. Anchor **A2**: `file · func · line`, the code as written (M4).
4. Ground **A3**: where the bytes live; the roofline position.
5. Pin **A4**: one runnable/measured number — asserted numbers don't exist here.
6. **VN Feynman block under every major EN block** — interleaved, never appended (08-11 law).
7. Worked **NEIGHBOR**, never the target (T3; the seal's teaching face).
8. **His prediction precedes every number**, including ones I already hold (T6/M3) — and it
   is **aimed at the load-bearing quantity of the lesson** (pretesting is item-specific:
   g=0.66 for the predicted item vs 0.01 for adjacent content — §10.8).
9. Hard concept → the **Feynman gate** (§10.4); ✅ only per the §10.5 bar, recorded durably.
10. Close with the **production/hiring linkage** + set one spaced callback for next session
    — interleaving ONLY between confusable variants (KL estimators, attention forms), never
    unrelated facts (§10.8).

### 10.8 The evidence base — verified verdicts (26/08 2-agent pass; meta-analyses cited)

**All 13 planks survive; none overclaimed.** STRONG: retrieval practice (Rowland 2014
g=0.50; Adesope 2017 g=0.61) · generation effect (Bertsch 2007 d=0.40) · self-explanation
(Bisra 2018 g=0.55) · fluency illusion (Bjork/Dunlosky/Kornell 2013). NUANCED, with the
boundary conditions now folded into the contract: **pretesting is item-specific** (2025
meta: g=0.66 predicted item vs 0.01 adjacent — aim the prediction at the lesson's
load-bearing number) · **interleaving** helps only for discriminating confusable
categories (Brunmair & Richter 2019; null-to-negative otherwise) · **drawing**: the
supported condition is exactly ours — learner draws AFTER tracing real code, transfer
benefit (Fiorella & Zhang 2018) · **worked-example + expertise-reversal**: high assistance
for novices d=0.505, and the asymmetry says over-assisting is the CHEAPER error → the
faded-scaffolding default is right (Tetzlaff 2025, n=5,924) · **working memory 3–5 chunks**
(Cowan 2001) — one-micro-concept pacing is a design extrapolation, labeled as such · **L1
elaboration** of L2-taught technical content is supported via cognitive load (Roussel/
Sweller 2017) — the VN layer's evidence base · **deliberate practice**: justify our reps
via retrieval/feedback/worked-example literatures, NOT Ericsson (explains only ~4% variance
in education, Macnamara 2014). **Both program-cited arXiv IDs are real and match**:
2507.05629 (73→89%, ~60-student quasi-experiment — cite as suggestive) and 2601.20245
(50% vs 67%, randomized, d=0.738, Anthropic-affiliated — [FACT]-grade). **Never cite Wang
& Fan 2025 (ChatGPT tutoring g=0.867) — RETRACTED.**

**The one open tension — recorded as data, decision is the operator's (T-loop order is
settled law):** productive-failure meta-analysis (Sinha & Kapur 2021, 53 studies, g=0.36)
finds a SHORT committed problem-solving attempt BEFORE instruction beats instruction-first
for conceptual understanding in adult learners — provided full instruction follows. What
the worked-example literature rules out is *prolonged unsupported* problem-solving, not a
brief committed guess. Compatible upgrade if the operator wants it: a **≤60-second
committed micro-guess before the T1 frame** (then teach fully, as now). The 08-14 "teach
me first" ordering stands until the operator says otherwise.

**The LLM-era finding — external validation of this repo's architecture [FACT]:** the
documented 2024–26 practitioner pattern is *hand-generate the core you are mastering
against a measured target; delegate everything around it; use the LLM as an on-demand
Socratic reference, never as ghostwriter of the thing being learned*. Karpathy's nanochat
is "basically entirely hand-written (with tab autocomplete)" — agents were "net unhelpful"
(X, 2025-10-21) — while by Jan 2026 he has Claude running/babysitting his nanochat
*experiments* end-to-end (X, 2026-01-03/05: "writes implementations, debugs them with toy
examples… launches training runs, babysits them by tailing logs"). Randomized AI-learning
studies agree the harm is conditional: guardrailed, cognitively-engaged AI use preserves
learning (Shen & Tamkin 2026; Bastani PNAS 2025). **That is exactly the sealed-four + L2 +
learn-mode architecture this repo runs. Keep it.**

## §11 · The NVIDIA lane — the reverse-engineered bar (researched 2026-08-27; 10-agent pass, every load-bearing claim adversarially re-verified: 37 CONFIRMED / 5 CORRECTED / 0 REFUTED)

> Method [FACT]: NVIDIA's board (2,663 live reqs) enumerated from `jobs.nvidia.com/careers/sitemap.xml`;
> full JD texts pulled from the official eightfold JSON API (`jobs.nvidia.com/api/apply/v2/jobs/<id>`);
> comp from levels.fyi + FY2026 H1B/LCA filings; interview loops from opened candidate reports (Blind/HN);
> tech axes from NVIDIA dev blogs / GitHub / LLVM source / GTC 2026. Verifiers re-opened every URL and
> re-grepped every quote. This section is the durable home; PLAN.md stays the only ordering authority.

### 11.1 The ten live postings (the highest-value kernel/perf surface, 27/08)

| Req | Role (team) | Base band [FACT] | Exp gate | What it actually demands |
|---|---|---|---|---|
| JR2018988 | **Sr SWE, CUTLASS Kernels** (Math Libs) | $152–241.5k L3 / $184–287.5k L4 | **3+ yrs** | "Write Tensor Core-based deep learning kernels such as grouped-GEMM, attention, and convolution using CUTLASS CUDA C++ and Python DSL **for Blackwell, Rubin, and future architectures**"; assembly-level; PREF: PTX/cuTile + "**Open-source contributions to math kernel libraries**" |
| JR2021962 | **Sr Inference Engineer, GPU Kernel Optimization** (LLM Inference Perf; updated 27/08) | $184–287.5k | 6+ yrs | "silicon-measured kernel benchmarking infrastructure"; "**agentic optimization systems that improve GPU kernels at the assembly layer**"; REQ: **agentic-AI-systems experience**, CUPTI/NSYS/NCU bottleneck attribution, TRT-LLM/SGLang/**vLLM**, read PTX/SASS; PREF: **FlashInfer/Triton/CUTLASS contributions** |
| JR1996103 | Sr DevTech Engineer — AI | $184–287.5k L4 / $224–356.5k L5 | 8+ yrs | in-depth CPU+GPU arch; low-level perf opt; publish blogs; "**influence the design of next-generation hardware architectures**" (= the HW/SW co-design seat) |
| JR2017806 | Sr Perf Compiler Engineer — Triton (open-source team) | $184–287.5k | 8+ yrs | MLIR on Triton DSL; "numerics (like block-scaled floating point)"; assembly hands-on |
| JR2014705 | Sr AI SWE, Kernel Libraries (attention kernels; FlashInfer/vLLM/SGLang orbit) | $184–287.5k | 6+ yrs | inference engines + kernel codegen; the ONLY post saying "**PhD are preferred**" |
| JR2013405 | Sr System SWE — Dynamo-Triton Server | $152–241.5k / $184–287.5k | 5+ yrs | **Rust** + C++; distributed serving; OSS-contribution literacy |
| JR2021665 | Sr System SWE, Agentic Inference — Dynamo | $224–431k L5/L6 | — | "develop open source software to serve inference"; PREF: "contributions to open-source AI inference frameworks (e.g., **vLLM**, TensorRT-LLM, SGLang)" |
| JR2017016 | **Principal High-Perf LLM Training Engineer** | **$272–431.25k** | **12+ yrs** | "**Demonstrated principal-level technical impact**"; DP/TP/PP/EP/SP; profiling/perf-modeling track record — the principal bar in writing |
| JR2020181 | Sr DL Performance Architect (arch/perf modeling; updated 27/08) | $152–241.5k / $184–287.5k | **4+ yrs** | analytical perf modeling + profiling — the lowest senior gate; our predict-the-number method IS this job |
| JR2016912 | Sr SWE — PyTorch & AI Frameworks | (unposted) | 8+ yrs | PyTorch core; PREF: ecosystem contributions, TensorRT/cuBLAS/cuDNN/NCCL |

(JR2007055 TensorRT-LLM Python-first, 4+ yrs, likely closed — vanished from the Workday board 27/08.
cuDNN-titled reqs: **zero live** in the sitemap; Kernel Libraries is the live attention-kernels team.)

### 11.2 The requirement axes, frequency-ranked over the 10 posts [FACT]

1. **Profiling / perf analysis (NSYS/NCU/CUPTI-grade)** — 10/10 (the #1 axis, matching the 81% root-cause finding of §2)
2. **Degree WITH escape hatch** — 10/10 write "(or equivalent experience)"; PhD never strictly required
3. CUDA / GPU parallel programming — 9/10 · GPU/computer architecture depth — 9/10
4. C++ (paired with Python) — 8/10 · Python first-class — 6/10
5. **Open-source contribution track record — 6/10** (named as THE stand-out; repos named: FlashInfer, Triton, CUTLASS, vLLM, TRT-LLM, SGLang)
6. Kernel authoring in CUTLASS/Triton/cuTile — 5/10 · LLM inference frameworks — 5/10
7. Numerics / low-precision — 4/10 · analytical perf modeling — 4/10 · ML compilers (MLIR) — 4/10
8. PTX/SASS reading fluency — 3/10 (the three most kernel-core roles)

### 11.3 Leveling + comp truth [FACT — Blind (confirmed), levels.fyi, H1B]

- IC ladder IC1–IC9: **IC3–IC5 all carry "Senior" titles; IC6 = Principal directly** (no staff tier);
  IC7 Distinguished (rare), IC8 Sr Distinguished (1–2 people), IC9 Fellow (one).
- TC (levels.fyi, 27/08): IC3 ≈ $318k · IC4 ≈ $378k · IC5 ≈ $467k · **IC6 Principal ≈ $608k avg
  ($533–760k+)** · IC7 ≈ $1.02M. Stock is the increment; bonus ≈ 0; negotiation = base + initial grant.
- Posted base bands are **standardized by level across orgs** (L3 $148/152–241.5k · L4 $184–287.5k ·
  L5 $224–356.5k · L6 ≈ $272–431k): org choice buys visibility/mandate, not cash.
- Org ranking for a kernels/linear-attention person [INFERENCE]: (1) TRT-LLM/FlashInfer-orbit inference
  (open, vLLM/SGLang-integrated, MLSys-award ecosystem) → (2) CUTLASS/math libs (deepest kernel
  prestige; feeder to arch + inference leadership: Thakkar, Chetlur) → (3) DevTech (elite, same bands,
  customer-facing, "half our time pure research").

### 11.4 The interview loop as documented [FACT — opened candidate reports]

- New-grad DevTech-AI (offer received): fit call → 1 h GPU basics → **two 1 h technicals + two 48 h
  take-homes (one parallel-CPU algo, one GPU algo)** → four 1 h finals ("here's an algorithm, how
  would you write it in parallel"). Senior DevTech: **12 interviews + 2 take-homes** documented (no offer).
- NOT LeetCode: HW–SW interaction, parallelize-this, memory coalescing with Nsight follow-ups, the
  GEMM ladder (naive ~1–5% cuBLAS → tiled → register-blocked 50–90%), occupancy/roofline [INFERENCE].
- Documented rejection pattern: fumbled coalescing follow-ups; knowing only FP32→INT8; failing probes
  "**two levels deeper** than the outcome" ("how did you know it was bandwidth-bound?") — verdict
  "optimized by trial and error rather than by rigorous measurement". **Our ledger discipline is the
  literal counter-training.** (Correction: "linear algebra + numerical methods" as explicit JD text is
  a 2010 listing; the current AI DevTech req JR2008156 does not list it.)

### 11.5 The PhD escape hatch — evidence it is exercised [FACT]

Every one of the 10 posts carries "(or equivalent experience)". Named no-PhD (or pre-PhD) paths into
exactly these seats: **Zihao Ye** hired as NVIDIA Senior Compiler Engineer while a 4th-year PhD
student — FlashInfer bought the seat before the degree; **Lei Mao** (two Masters, public CUDA/TensorRT
blog) in DL-perf-adjacent NVIDIA work; **Horace He** (no PhD; torch.compile/FlexAttention →
Thinking Machines, author of the batch-invariance post behind vLLM #42960 — OUR lane's origin);
**Simon Boehm** (MSc; the canonical CUDA matmul worklog → Astera; Anthropic move widely repeated
[UNCERTAIN]). An NVIDIA hiring manager's own senior/principal DevTech pitch: "seasoned professional
engineer" + "excellent communications" — no degree language.

### 11.6 The OSS→NVIDIA pipeline — with the negative finding [FACT]

- NVIDIA JDs name contributions to **FlashInfer/Triton/CUTLASS/vLLM/TRT-LLM/SGLang** as the stand-out.
  NVIDIA routes its best inference kernels INTO FlashInfer (Ye, TQ Chen, Ceze VP, Grover); the NVIDIA
  CCCL team competes on the GPU MODE leaderboard under their own names and invites contact.
- GPU MODE × NVIDIA "Blackwell NVFP4 Kernel Hackathon" (Nov 10 2025 – Feb 13 2026, B200): four
  problems (NVFP4 batched-GEMV · GEMM · gated dual GEMM · grouped GEMM); prizes DGX Spark / RTX 5090 /
  RTX 5080 + GB300 grand. **NEGATIVE FINDING: no public recruiting promise, no public winner-hire
  story — the pipeline is implicit (GTC ceremony, engineers in the Discord), vs Saroufim's explicit
  "hire the cracked engineers" framing.** (This corrects our earlier "winners recruited" note.)
- **Live next window: MLSys 2026 FlashInfer AI Kernel Generation Contest — NVIDIA Track**
  (mlsys26.flashinfer.ai; NVIDIA+MLSys+FlashInfer+Modal; DGX Spark / 5090 / 5080) — an AI-kernel-
  generation contest, i.e. exactly this repo's co-lab premise (agents generate, human verifies on
  silicon).

### 11.7 NVIDIA's 2026 technical axes — what the platform telegraphs [FACT unless tagged]

- **Python-first CUDA**: CUDA Tile IR + cuTile Python (13.1, Dec 2025; CC 8.x/10.x/11.x/**12.x — the
  standing 5090 is supported**; repo open-sourced but not accepting external PRs), CUDA 13.2 (Nsight
  Python), CuTe-DSL beta → graduation ~end-summer 2026; CUTLASS 4.8.0 targets SM90/100/103/**120**.
- **The chasm**: FA4 is written in CuTe-DSL Python (1605 TFLOP/s BF16 on B200, 71% util; 1.1–1.3× vs
  cuDNN and 2.1–2.7× vs Triton — **both forward-pass numbers**, correction logged); PyTorch: the
  Triton-vs-hand-tuned gap "has grown to a chasm" on Blackwell (tcgen05 async + TMEM 256 KB/SM not
  expressible in Triton). The scarce engineer bridges DSL ↔ microarchitecture ↔ numerics ↔ serving
  [INFERENCE, structurally supported].
- **NVFP4 is the strategic numerics bet**: production training (JAX/MaxText: 1.31–1.73× vs FP8, loss
  within 0.026 nats) + inference (≤1% degradation on DS-R1) + an active pitfall literature (Four Over
  Six: near-max-value block-scaling error). Numerics-of-4-bit is a hiring axis, not a niche.
- **Rubin**: 336B transistors, 50 PF NVFP4, 288 GB HBM4 @ 22 TB/s, NVLink6; **Rubin CPX = a prefill
  GPU** (disaggregation baked into silicon, end-2026); sm_107 staged in LLVM for CUDA 13.4/PTX 9.4 —
  **no released toolchain can target it yet** (13.3u1/PTX 9.3 current). Cadence: Rubin '26 → Ultra '27
  → Feynman '28 (Huang, Dwarkesh 4/2026). Roofline+numerics skills transfer; incantations don't.
- **AI-writes-kernels is official**: Huang — "We use a ton of AI to create the kernels that we have";
  JR2021962 REQUIRES agentic-systems experience and calls the job "validate findings with rigorous
  silicon measurements". The sealed-four + L2 + verifier architecture of this repo is that role,
  rehearsed daily.

### 11.8 The overlay N1–N6 — slots into PLAN.md; changes NO ordering

- **N1 · FlashInfer = NVIDIA-legible surface #1** (already the sm_120 NVFP4 backlog lane, #2577):
  NVIDIA absorbed the project and its JDs name it first — the vLLM lane and the NVIDIA lane are the
  same work here. Fires per PLAN.md (after the harness claim).
- **N2 · CuTe-DSL/cuTile rung on the 5090** ($0): re-express one owned kernel (GEMM receipt or the
  K2 chunk kernel) in CuTe-DSL; narrate Triton-vs-CuTeDSL with measured numbers. Earns JR2018988's
  "CUTLASS C++ **and Python DSL**" + "PTX or CUDA/cuTile" preferred axis. Attach to M2 (§6).
- **N3 · NVFP4 numerics rung** (sm_120 has block-scaled FP4 mma): E001-style divergence analysis of
  E2M1+E4M3 block scaling; reproduce the near-max-value pitfall at micro scale; flashinfer #2577.
  Our numerics moat pointed at NVIDIA's flagship format.
- **N4 · MLSys 2026 FlashInfer contest (NVIDIA Track)**: enter with the agent+profiler loop (§6)
  once E1 + review have landed — it is the co-lab thesis, scored publicly.
- **N5 · Blackwell evidence day** (backlog, unchanged): the wgmma→tcgen05 narration with numbers on
  both is precisely the §11.7 chasm story.
- **N6 · NVIDIA application batch** (fires with rule 5, E1 public): JR2020181 (4+ yrs, perf modeling
  = our method), JR2018988 (3+ yrs, CUTLASS), JR2021962 (6+ yrs — the bullseye; apply anyway: every
  gate has the equivalence clause), + new-grad/intern DevTech pipeline. Interview prep = the existing
  reps: GEMM ladder, coalescing narration, 48 h take-home rehearsal, "two levels deeper" ledger drills.

### 11.9 Corrections logged by this pass (claims honesty)

1. "NVFP4 hackathon winners recruited by NVIDIA" → **no public evidence**; implicit pipeline only.
2. FA4 "2.1–2.7× vs Triton" is the **forward** pass (not backward); cuDNN incorporation confirmed for
   9.13/9.14 only.
3. "Math is an explicit DevTech JD requirement" → true in a 2010 listing; absent from the current req.
4. TensorRT-LLM JR2007055 likely closed (gone from the board 27/08).
5. Rubin sm_107 mapping = [INFERENCE] (LLVM staging), not an NVIDIA statement.

## §12 · The curriculum binding — one object, six altitudes (29/08, operator-set)

> **What this section settles.** The operator asked for the Vizuara curriculum to be followed
> exactly, mastered before building, and mapped to production tasks. §12.1 shows those are not
> three requests but one, because their best unit already *is* our open node. §12.2 is the
> depth the curriculum stops short of — the object at every altitude down to the ISA. §12.3 is
> the 1:1 ladder. **PLAN.md still owns ordering; this section adds no rung and no date.**

### 12.1 "Learn first, then build" — resolved, not overridden

The instruction is **true at the rung** and **false at the program**, and the distinction is
load-bearing:

- **At the rung it is already law.** Derive before you type; predict before you measure; a
  number you did not predict teaches nothing (T-loop · rule 2). Nothing here relaxes that.
- **At the program it is the named failure mode.** Jul–Aug 2026: ~87 planning files, 0
  kernels (rule 7). Nine weeks of curriculum before the first shipped artifact reproduces it
  with a better syllabus.
- **The conflict dissolves on inspection**: their Session 2 (*"How fast can this go? — roofline
  lab, predict then measure"*) **is** `MASTERY_LEDGER.md` Row 000 D1. Their capstone (*"a real
  kernel problem, graded at demo day"*) **is** the #45819 adjudicating review. **The
  curriculum's own best units are the production nodes.** So the binding rule is:

> **A curriculum unit with no production node attached does not run.** Units that map to a
> live node run *as* that node. Units already receipted are not re-run. Units gated by
> hardware wait for the rental that funds them. There is no third category.

### 12.2 The object at six altitudes — below where the curriculum stops

The whole program is **one object**: `S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`,
`o_t = q_tᵀ S_t`. Every rung is the same equation at a lower altitude. Vizuara teaches L0–L3
across nine weeks on *generic* kernels; L4–L5 on *this* recurrence is where the defensible
contribution is, and no course covers it because the object is 18 months old.

| | Altitude | The question | Where it lands here |
|---|---|---|---|
| **L0** | Algebra | Associative memory: write `k→v`, erase along `k`, decay by `α`. Widrow-Hoff delta rule; the eraser `(I−βkkᵀ)` is an orthogonal projection exactly when `‖k‖=1, β=1` — which is why the input path L2-normalises `k` | `paths.py:5-15` · K2 `core/kda.py` |
| **L1** | Two paths | Recurrent: `O(dk·dv)` state, **serial in T**. Chunked: parallel over chunks, but the product `∏(I−β_i k_i k_iᵀ)` across a chunk must be collapsed into a compact form — and *that* collapse introduces an operation with no counterpart in the recurrent path | Row 000 **D2/D3 — open, his** |
| **L2** | Numerics | Error is a **dynamical system**, not a per-element residual: `E_t = A_t E_{t−1} + δ_t` with `A_t = α_t(I−β_t k_t k_tᵀ)`. Because the state is *carried*, chunk 0's rounding is chunk 1's **input**. `‖A_t‖ ≤ α_t · max(1,\|1−β_t\|)` — so the gate is the contraction factor of the error recursion, not just of the signal | **E001** · Row 001 predictions — **open, his** |
| **L3** | Kernel | Chunk 64 (`FLA_CHUNK_SIZE`); the `(dk,dv)` state at 128×128 fp32 = 64 KB — too big for registers, fits sm_120's 99 KiB smem, does not fit sm_100's assumptions the same way; tile config `(BT,BK,BV)`; roofline position | M1/M2 · E2 |
| **L4** | **Dispatch** | **Where batch-invariance actually breaks — MEASURED 29/08, see the narrowed claim below.** `prepare_chunk_indices(cu_seqlens, BT)` (`fla/ops/utils/index.py:156-164`) maps global chunk → (seq, local chunk) and sets the **grid** via `NT = len(chunk_indices)` (`gate.py:177`): change `cu_seqlens` → different CTA count and chunk→CTA mapping, same per-CTA tile shape. Batch size reaches numerics through exactly one door — the **generic gate cumsum** (`cumsum.py:29`, `key=['B',…]`), taken by default at `chunk.py:63`. Three further forks: tf32-vs-ieee on the solve, vendor warp lists, live autotune | **the review · E3** |
| **L5** | Bits | FMA contraction (`a*b+c` = one rounding vs two — `-fmad=false` is a checkable knob); `mma.sync` accumulates fp32 in a **hardware-fixed intra-instruction order**; `__shfl_down_sync` butterflies are fixed for fixed warp count; `atomicAdd` is genuinely order-nondeterministic; fp32 addition is non-associative | N2/N3 · B200 day |

**The Principal-level claim this ladder produces** — **NARROWED 29/08 by the Triton read it
named as its own test.** Source: `~/Desktop/oss/fla` @ `c3db408` (HEAD 29/08), first-party.

*Unchanged and now better supported:* **the nondeterminism is not in the silicon.** Tensor-core
accumulation order is fixed by the instruction encoding; warp-shuffle reduction trees are fixed
by warp count; both are bit-reproducible at fixed configuration. The fix is **pinning the
reduction strategy**, not rewriting math.

*Corrected — the previous mechanism ("autotune picks a different `(BT,BK,BV)` when `cu_seqlens`
changes") is REFUTED as stated.* Every autotune key on the state/WY path excludes batch and T,
and `T` is explicitly de-specialized:

| Kernel | `key=` | batch in key? |
|---|---|---|
| `chunk_fwd.py:36` kkt+solve fused | `['H','HV','K','BC']` | no |
| `wy_fast.py:39,119` recompute w/u | `['H','HV','K','V','BT','BK','BV','IS_VARLEN']` | no |
| `solve_tril.py:34,104,194` | `['BT']` · `['H','BT','IS_VARLEN']` | no |
| `chunk_scaled_dot_kkt.py:29` | `['H','HV','K','BT','IS_VARLEN']` | no |
| `gate.py:59,117,240` fused gate cumsum | `['H','BT','IS_VARLEN','REVERSE']` · `['H','BT']` · `['H']` | no |
| **`cumsum.py:29,81` generic local cumsum** | **`['B','H','BT','IS_VARLEN','REVERSE']`** | **YES** |

`@triton.jit(do_not_specialize=['T'])` on all of them (e.g. `chunk_fwd.py:39`, `wy_fast.py:42`).

**The one real batch-size→numerics path, and it is a two-line branch** (`chunk.py:51-69`):
`use_gate_in_kernel=True` → `gdn_gate_chunk_cumsum` (batch-independent key) · **`False` — the
default (`chunk.py:46`) → `chunk_local_cumsum`, whose key contains `B`.** So on the default
path, changing batch size changes the autotune key of the **gate prefix-sum**, which can select
a different `num_warps` — and `num_warps` sets the warp-shuffle reduction tree, i.e. the
**summation order of the log-gates**. Different order → different rounding in `g` → different
`a_t = exp(cumsum)` → different `β_t/a_t` entering the WY solve. Batch size reaches the numerics
through the *gate*, never through the state kernels.

**Three further dispatch forks found in the same read (all [FACT], all `file:line`):**

- **tf32 on the triangular solve** — `chunk_fwd.py:20-23`: `SOLVE_TRIL_DOT_PRECISION =
  tl.constexpr('tf32')` if `IS_TF32_SUPPORTED` else `'ieee'`, where `IS_TF32_SUPPORTED =
  IS_NVIDIA and capability[0] >= 8` (`fla/utils/_device.py:152`). An **import-time** branch on
  compute capability setting the dot precision of the fused KKᵀ+`solve_tril` kernel — the most
  ill-conditioned step in the algorithm (our H1). **10-bit vs 24-bit mantissa, chosen by
  hardware, invisible at the call site.** This explains #45819's sm_120/sm_86/sm_90 regime split
  more directly than autotune does, and E001 can price it on CPU for $0 via `paths.py`'s
  existing `solve_dtype`.
- **Vendor changes the search space** — `wy_fast.py:26`: `RECOMPUTE_W_U_NUM_WARPS =
  [2,4,8,16] if IS_INTEL else [2,4,8]` (`_device.py:134`).
- **The default is LIVE autotune** — `cache.py:52`: `FLA_CACHE_MODE` unset → `DISABLED` →
  "always fall back to Triton autotune", and **`fla/configs/` is not in the repo** (the per-GPU
  JSON cache is opt-in, unshipped). Config choice is therefore a **runtime benchmark**: on a
  shared or noisy GPU two processes can select different configs from identical inputs. This is
  run-to-run nondeterminism that **requires no shape change at all** — a stronger and more
  easily demonstrated source than shape dispatch.

**Status:** the Triton half of this claim's test is DONE. The E1 half is not — the numbers that
price each fork are still unmeasured. Do not present any of the four as quantified.

**The error-recursion framework, stated on the neighbor** (the target case is the operator's
to derive — Row 001 is unwritten): for a scalar recursion `e_t = a·e_{t−1} + δ`, `|a|<1` gives
a geometric series saturating at `δ/(1−|a|)`; `|a|=1` gives a random walk whose magnitude
grows like `√T`. Two regimes, one parameter. **Which regime each cell of Row 001 sits in — and
therefore the ordering of its three predictions — follows from computing `‖A_t‖` for the real
operator. That computation is not performed here by design.**

### 12.3 The merged ladder — their 25 units → production nodes

Legend: **▶** live now · **✅** receipted (not re-run) · **🎯** slot open, $0 · **🔒** hardware-gated.

| Their unit | Our node | Consumer | State |
|---|---|---|---|
| S1 CPU parallelism | — (zero JD pull, §2) | — | skip |
| **L1 roofline predict-then-measure** | **Row 000 D1** | E1 → the review | **▶** |
| L2 CUDA model · PTX/SASS · GPU Puzzles | PTX fluency (gap 6) rides **N2** | NVIDIA JR2018988 | 🎯 |
| L3 memory hierarchy · transpose · ncu | `kernels/` ✅ + **Proton lab** | profile gap 2 | ✅🎯 |
| L4–L5 GEMM → 93.7% | re-own **≥90%** (hold 81.9%) | interview rep | 🎯 |
| L6 tensor cores / WMMA | mma.sync ✅ + **N2 CuTe-DSL** | JR2018988 "Python DSL" | ✅🎯 |
| L7 profiling · 3 sabotaged kernels | self-sabotage drill (poached) | root-cause rep (#1 axis) | 🎯 |
| L8 attention · FA1 | FA2 ✅ (50% SDPA, 44× mem) | — | ✅ |
| DD1 FA2/FA3 | FA3 async | H100 half-day | 🔒 |
| DD2 beat cuBLAS on H100 | TMA/wgmma ladder | prelude to N5 | 🔒 |
| DD3 Triton→CUTLASS→CuTe | **N2** + Triton→Gluon rung | JR2018988 · JR2017806 | 🎯 |
| DD4 serving kernels | **land inside vLLM** — not a lab | maintainers | **▶** |
| DD5 Blackwell / NVFP4 | **N3** sm_120 FP4 ($0) + flashinfer #2577 | JR2021962 · gap 4 | 🎯🔒 |
| DD6 FA4 | read the CuTe-DSL source ($0); numbers | B200 day | 🔒 |
| DD7 FlashMLA / DeepGEMM | reverse-engineering loop §10.3 | taste rep | 🎯 |
| DD8 LLM kernels · agent+profiler loop | **the verifier — our capstone, their open question** | **Wafer** · Anthropic Perf-RL · JR2021962 | 🎯 |
| **Capstone** (Crusoe, graded at demo day) | **E1 map → the #45819 review → E3** | vLLM maintainers | **▶** |

**Their two structural advantages we do not have, named honestly:** a cohort (peer pressure +
a fixed clock) and a graded deadline. **Our substitutes:** the PR thread is the demo day, and
the maintainer is a harsher grader than any instructor. **Their structural disadvantage:**
their capstone is retired at demo day; ours stays in production and keeps paying.

### 12.4 Their four pillars, applied exactly — plus the fourth field

| Their pillar | Applied here | What we add |
|---|---|---|
| *"Modern — 2025-26 frontier, first principles, why/where/who for every topic"* | §3 stack ledger + §2 live market scan, both date-stamped | **the receipt** — `file:line` or a `bench/RESULTS.md` row. Their card has 3 fields; ours has 4, and the 4th is the whole difference between a map and territory |
| *"Explained simply — the idea in plain words before the jargon"* | A0 frame opens every turn; the **VN Feynman block** is the plain-words test (§10.4) | if the VN can't carry it, the concept is not owned — a *gate*, not a nicety |
| *"Built by hand, live — you write the kernels yourself"* | the **sealed four** (rule 3) is this rule with teeth: agents are locked out of the oracle, the harness, RL loss math, verifier tolerances | they hand-build *in* a session; we hand-build what a maintainer will interrogate |
| *"Job-ready — maps 1:1 to what Wafer / NVIDIA / top labs hire for"* | §11 (10 NVIDIA JDs full-text) · §2 (N=21) · §9.4 (the hired-from-OSS pattern) | **their own named example is now a lane**: Wafer is YC S25, ~6 people, and maintains a 2.2k★ public resources repo with a fillable determinism gap — see PLAN.md backlog |
