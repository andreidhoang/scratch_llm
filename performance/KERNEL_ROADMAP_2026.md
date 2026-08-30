# Production Kernel Engineering — the 2026 Roadmap (verified 2026-07-14)

> **What this is.** The kernel-engineering operating spec: (a) a dissection of the Vizuara Kernel
> Engineering Workshop curriculum against this repo's ledger, (b) a five-stream, primary-source-verified
> re-map of the mid-2026 kernel frontier (supersedes the 2026-07-09 §6 note where they conflict),
> (c) the **embed-vs-separate verdict**, and (d) the phased roadmap + mastery protocol that runs
> inside `MASTER_PLAN_2026-07-12`'s arbitration — it does **not** reopen the kernel lane early.
>
> **Standing-rule accounting.** MASTER_PLAN bans new planning .md until a hardware-stamped number
> exists. This doc is the explicitly user-requested exception and consumes the rationed plan-note
> allowance. It is the **last** kernel planning doc: everything downstream of it is ledger entries
> (`bench/RESULTS.md`, `MASTERY_LEDGER`), postmortems, or code.
>
> **Provenance.** Synthesized 2026-07-14 from five parallel deep-research streams (vendor/hardware ·

> ⚠️ **ERRATA-D (+ REVISION 1) — stamped 2026-08-14.** This document was **not consulted** when ERRATA-D
> was written, and it should have been. Its §3 already held a session-by-session Vizuara dissection
> against the measured ledger, dated 2026-07-14, with the verdict *"strictly dominated — skip; poach the
> sabotage drill, the CPU session, and the book packaging."* A market-side pass on 2026-08-14 (51 postings,
> live syllabus re-extraction, field-state survey) reached the same verdict independently. **This §3
> verdict stands and is not superseded.** Full list: the ERRATA-D block in
> `Desktop/plan/KERNEL_RL_MASTERY_EXECUTION_2026.md`; citations in memory `demand_surface_2026`.
>
> **What ERRATA-D adds to this document — evidence, and four deltas:**
>
> - **§2's "recurring JD nouns" now has counts** (45 postings with full bullet text): profiling/root-cause
>   **38/45** · Python+C++/Rust **30** · CUDA C++ **22** · TP/PP/EP/CP/FSDP **18** · low precision **17** ·
>   vLLM/SGLang internals **13** (named as a *pair* in 9) · NCCL **12** · **RL infra bundled with perf 11** ·
>   Triton **11** · TPU/XLA **11** · determinism/numerics **10** · CUTLASS/CuTe **7** · **ROCm 3** ·
>   **Pallas 1** · hand-written GEMM **0** · sparse/long-context attention **0** · "batch invariance" **0**.
>   §2's instinct was right on every noun; the counts change only the *ordering* of effort.
> - **§3's omissions list anticipated ERRATA-D · D7 and D8 by a month** — *"determinism &
>   batch-invariance… now a vLLM/SGLang product feature and an RL-correctness lever"*, *"perf-regression CI
>   & kernel packaging"*, *"multi-platform hedge (ROCm/HIP, Pallas/NKI)"*. **Confirmed with evidence:**
>   the Anthropic *Research Engineer, Performance RL* req puts **ROCm and Pallas in the *required* block**;
>   and **zero postings use the words "batch invariance"** while 10/45 demand it as *numerics · correctness
>   · reproducibility · **regression detection***. So the hedge is a checkbox, and the determinism work has
>   to be **translated on contact**, never named by its mechanism first.
> - **§4's authoring assumption flips (D5).** FlashAttention-4 is written in **CuTe DSL**, not CUDA C++;
>   PyTorch 2.13 ships a CuTeDSL Inductor backend; CUTLASS 4.7.0 added a Primitives API *beneath* CuTe.
>   **P-phase authoring targets Triton first, Gluon or CuTe DSL second**; CUDA C++ is for reading. This
>   sharpens §3's "IV.d3 = P2" rather than reopening it.
> - **§4's P1/P2 rental framing needs one correction and one release.** Correction: vast.ai has **zero
>   VM-mode offers for H100/H200/B200/B300/A100** — but per memory `hardware_truth`, **root on any real VM
>   or bare metal is always granted `ncu` counters**, so P1 is bookable at **Latitude.sh $1.68–1.99 bare
>   metal / Hyperstack $2.50 / Crusoe $3.90** (Crusoe ships the only first-party ncu how-to). The
>   ERR_NVGPUCTRPERM debt in §3's I.3 row is therefore payable now. Release: **sm_120 has native
>   block-scaled FP4** (`mma.sync.m16n8k64.mxf4nvf4`) — which this repo's own A5 R3 row already
>   demonstrated on 2026-07-04 (**NVFP4 20.43 dB vs MXFP4 18.74 dB = 1.48×**). Only NVFP4-**MMA
>   throughput** stays B200-gated, exactly as §7 says.
>
> **And the correction that runs the other way — ERRATA-D was wrong and this repo proved it.**
> ERRATA-D claimed the plan lacked analytical performance modelling, multi-GPU collectives, and host-side
> overhead work. All three were **already here**: `utils/comms_calc.py` (~25 closed-form functions) +
> `memory_math.py`; `ddp.py`/`zero1.py`/`fsdp.py` with **767 lines of green tests** and the written
> `deploy/runbooks/A2_multigpu_nccl_bench.md`; and the A1 R1 ledger row whose root cause is *"955 kernel
> launches/token + a `.item()` host-sync — 84% of the wall."* The revised, much narrower gaps are:
> **cost-to-serve** ($/1M tokens from microbenchmarks), **running A2** rather than building it, and the
> **RL-loop / Python-GIL** half of host-side work. **The durable rule: before writing "the plan lacks X",
> grep the repo for X.** This corpus has now made that mistake twice — §3.3 once, ERRATA-D once.
>
> Two items from §3 that ERRATA-D endorses and the Desktop corpus does not carry: **the "debug 3 sabotaged
> kernels" drill** — §3 calls it the best idea in the syllabus, and it is the cheapest possible rehearsal
> for the live optimize-this-kernel round — and **the marketing fact-check habit** (FA4 *"first attention
> kernel to break a petaflop"* is false as stated; FA3 FP8 hit ~1.2 PF on H100 in 2024). Keep both.
> attention/GEMM frontier · serving/MoE/distributed · AI-written kernels/DSLs · hiring), each returning
> dated primary sources tagged [VERIFIED]/[REPORTED]. Claims here carry the repo's [FACT]/[INFERENCE]
> convention; a [FACT] below means a primary source was fetched or cross-corroborated on 2026-07-14.

---

## 0. Verdicts

**V1 — Embed. Do not start a separate learning project.** The kernel curriculum stays in
`performance/` where ~70% of it is already **built and measured**. A fifth planning apparatus is this
portfolio's documented failure mode ("four world-class planning apparatuses and approximately zero
experiments" — MASTER_PLAN §1). What leaves the repo is **exit artifacts only** (§5): upstream PRs,
the DELTA kernel as a small public repo or FLA PR, and the worklog serialized publicly. Learn embedded;
publish extracted; the hiring narrative stays in `JOB_SPRINT`.

**V2 — Skip the $3,000 Vizuara workshop; strip-mine its syllabus for free.** Session-by-session (§3),
this repo's ledger already contains measured superiors of Parts I–III and most of IV–V. The genuinely
new remainder — FA4/Blackwell depth, an AI-written-kernels module, a graded capstone — costs ≈ $150–400
of rental days plus two roadmap modules (§4 P2, M-AI), and the "door to the frontier" it sells (the SF
GPU-startup capstone, almost certainly Wafer — a 4-person, $4M-seed YC company [FACT]) is a door you
can knock on directly with a better artifact. Its late-September start also lands *inside* your
application window. Joining the free waitlist for optionality is fine; **gate nothing on it.**

**V3 — Timing (rebalanced 2026-07-14 evening, user-directed).** The kernel lane runs a daily
**4–5 h block** paired with the RL flagship block (JOB_SPRINT §8.1): the 30–45' quarry rep opens the
block, deep work follows — M-AI → **P1 H100 ≈ Jul 18–19** → **P2 B200 ≈ Jul 22–25** → **DELTA from
≈ Aug 10** (pulled forward from Week 10+; niche re-check at open). Flagship stays senior: if the day's
CRL node slips, deep work yields, the quarry rep survives. You are still your own founding cohort —
two months before Vizuara's starts.

**V4 — Flip the kernel lane to `learn` mode.** The binding constraint is not missing content; it is
the mastery deficit (4/89 teach-backs) against **AI-prohibited interview loops** (MASTER_PLAN §2.3).
`.claude/execution-mode` already implements the contract (ADR-0013): for kernel-lane work the agent
scaffolds, sabotages, profiles, quizzes — and never types the kernel body (§6).

**V5 — DELTA holds as the #1 artifact; the niche was re-verified open on 2026-07-14.** No public
library ships an FP8/NVFP4 **recurrent-state decode** kernel for GDN-class linear attention [FACT,
negative-result search]. Adjacent demand is loud: Qwen ships FlashQLA (GDN chunked-prefill, TileLang,
2026-04) [FACT]; FlashInfer's MLSys-2026 kernel-generation contest ran a **Qwen3-Next Gated DeltaNet
track** [FACT]; GDN's author (Songlin Yang) is now at Thinking Machines [FACT]. Architecture↔kernel
co-design (DELTA married to F10) remains the scarcest signal.

---

## 1. Frontier delta — what changed or got confirmed since the 2026-07-09 note

> **2026-07-30 re-verification (maintenance-contract pass, primary-checked).** Deltas since 07-14:
> **(a)** CUTLASS **4.5.2 → 4.6.0** (changelog 2026-07-09; CuTe DSL adds CUDA 13.1 support) — the P2 CuTe-DSL rung targets 4.6.
> **(b)** vLLM **v0.26.0** shipped (2026-07) and the **AFD plugin** (2026-07-23) productizes Step3-style **Attention–FFN disaggregation** — a new serving-architecture line beyond PD-disagg (added to A1 §4.6 + references).
> **(c)** `flash-attn-4` is pip-installable (JIT, Hopper+Blackwell, `cu13` extra); FP8/FP4 FA4 numbers **still** not public; current cuDNN (9.24) matches FA4 (techniques adopted, not merged); the Hopper-decode regression stands.
> **(d)** Hardware: **B300 = `sm_103`** (CUDA 12.9+, `compute_100f` family-compatible), GB300 NVL72 shipping (~14–15 PF dense FP4, 288 GB, FP64 ~1.2 TF); NVIDIA's **Vera Rubin** page is live (VR200: 288 GB HBM4 @ 22 TB/s, NVLink-6 3.6 TB/s, ~50 PF NVFP4 vendor figure; volume H2-2026). A B300 may substitute the P2 B200 rental at equal price — same ISA contract, +50% FP4.
> **(e)** DELTA niche: **still open** (spot-checked 2026-07-30 — FLA active at standard precision only, e.g. chunked-GDR B300 bug fla-org/flash-linear-attention#945; FlashInfer PRs; community repos — no public FP8/NVFP4 recurrent-state GDN decode kernel). Demand signal strengthened: Qwen3.5 / GLM-5 / Nemotron all ship GDN-or-hybrid layers in production.
> Curriculum files (00_foundations, A1–A7, references, PERF_ENGINEERING_SPEC) refreshed accordingly — see git diff 2026-07-30.

Corrections first, then fresh anchors. Each line: claim → status → source (dated).

| # | 2026-07-09 note said | 2026-07-14 verified state |
|---|---|---|
| 1 | FA4 = arXiv 2603.05451, CuTe-DSL, 1605 TF/s B200 / 71% | **CONFIRMED** [FACT]. Paper 2026-03-05 (Zadouri, Hoehnerbach, Shah, Thakkar, Dao et al.); code public in `Dao-AILab/flash-attention` under `flash_attn/cute/`, pip `flash-attn-4` (JIT, no nvcc). Blog says 1605, abstract 1613 TF/s — cite "≈1.6 PF, 71%". Techniques verified: 1 Load + 1 MMA + 8 Softmax + 4 Correction warps; **conditional/lazy softmax rescaling** (~10× fewer rescales); hybrid exp = MUFU.EX2 + cubic-Horner software exp2 on FMA cores; bwd 2-CTA MMA halves atomic traffic. **Caveats:** cuDNN 9.13+ *adopted the techniques* (not a code merge) and now matches FA4 per Tri Dao himself; **FA4 decode on Hopper regresses vs FA3 at long seq (−49% @16K, no SplitKV)** — FA4 is not a strict upgrade. (tridao.me/blog/2026/flash4 · arxiv 2603.05451 · modal.com reverse-engineering post · SGLang docs) |
| 2 | Headline kernel = FA4-class on Blackwell in CuTe DSL | **CONFIRMED and strengthened**: PyTorch **2.13 (GA 2026-07-08) shipped a CuTeDSL Inductor backend** alongside Triton [FACT]; CUTLASS at **4.5.2** (2026-06-16), CuTe DSL pip `nvidia-cutlass-dsl`, Beta → GA "end of summer 2026"; NVIDIA claims PTX/SASS parity with C++ CUTLASS. (pytorch.org 2.13 blog · github.com/NVIDIA/cutlass releases) |
| 3 | Add a CuTe-DSL / TileLang authoring rung | **CONFIRMED**, with one correction: **Qwen FlashQLA is TileLang, not CuTe-DSL** [FACT] (github.com/QwenLM/FlashQLA, ~2026-04-29; claims 2–3× fwd / 2× bwd over FLA's Triton GDN chunked kernel, SM90+). TileLang's other production proof: DeepSeek ships DSA kernels in TileLang + CUDA. DSL verdict mid-2026: **Triton = default substrate · CuTe-DSL = beat-cuBLAS/cuDNN tier · TileLang = novel-op research→prod tier**; Gluon/TLX = OpenAI/Meta-internal power tools; Helion in beta/1.0 and a `vllm[helion]` extra; **cuTile shipped stable** (CUDA 13.2 Python, 13.3 C++) but ecosystem is young. |
| 4 | tcgen05/NVFP4 GEMM vendor-served by CUTLASS → understand+use | **CONFIRMED** [FACT]: CUTLASS 4.x ships SM100 dense/grouped GEMM, FMHA (ex. 77), MLA fused-reduction decode (4.2.0), GQA low-latency decode + paged KV (4.4/4.5.1), even SSD/Mamba kernels (ex. 111/112). **New reference worklog: ThunderKittens 2.0 (2026-02-19)** — BF16/MXFP8/NVFP4 B200 GEMMs matching/beating cuBLAS, with the best public tcgen05/PTX-memory-model writeup (the `tcgen05.cp` doc-typo that cost ~500 TF/s; `elect.sync`; cluster-occupancy quirks). Consumer sm120 ≠ datacenter sm100: **no TMEM/tcgen05 on your standing box** — Blackwell-datacenter reps are rental-gated by ISA, as already planned. (hazyresearch TK-2 post · CUTLASS releases · 0xsero Blackwell wiki) |
| 5 | A6 = wide-EP stack (DeepEP/EPLB/DBO/PD-disagg/FULL_AND_PIECEWISE) | **CONFIRMED as the shipped default** [FACT]: vLLM v0.25 (2026-07-11) — V0 deleted, FULL_AND_PIECEWISE default, DeepGEMM on by default, DeepEP-backed wide-EP (2.2k tok/s/H200 on the Dec-2025 large-scale-serving bench), `--enable-dbo`, NIXL PD-disagg (extended to hybrid-SSM models 2026-04); SGLang 0.5.15; NVIDIA **Dynamo 1.0 GA** (GTC 2026). New training-side comm layer to know: **NCCL 2.28+ device API (LSA/Multimem/GIN) + symmetric memory**, NVSHMEM 3.7, TorchTitan async-TP; **TorchTitan MXFP8+DeepEP = +41% DeepSeek-V3-style pretraining on B200** [FACT]. |
| 6 | DeepSeek-V4-Flash (285B) fresher optional target | **CONFIRMED shipped** [FACT]: V4 preview 2026-04-24 — V4-Pro 1.6T/49B-active, **V4-Flash 284B/13B-active** (not 285B), 1M context, MoE experts in FP4/rest FP8, arXiv 2606.19348; new attention = CSA/HCA compression + DSA top-k + mHC + Muon. vLLM day-0 post details the kernel surface. **R2 still does not exist.** Also new: **DSpark** spec-decode (2026-06-27, +57–85% over MTP-1). |
| 7 | DELTA re-scoped to fp8/nvfp4 recurrent-state decode, vs FLA baseline | **NICHE RE-VERIFIED OPEN** (2026-07-14): FLA ships `fused_recurrent_gated_delta_rule` (BF16-class); FlashQLA covers chunked-prefill only; no FP8/NVFP4 recurrent-state decode kernel found in FLA, FlashQLA, FlashInfer, TK, DeepGEMM, or vendor libs [FACT, absence-of-evidence]. Demand signals: FlashInfer MLSys-26 contest GDN track; Kimi Linear/KDA (2510.26692) ships channel-wise-gate GDN variant; Mamba-3 at ICLR 2026. **Risk unchanged:** a lab or FLA can close this any week — build it at Week 10+, not Week 20. |
| 8 | (not in note) | **AI-written kernels became a hiring line-item.** OpenAI JD "Kernel Performance & AI Tooling" requires agentic-workflow familiarity [REPORTED]; NVIDIA CompileIQ (CUDA 13.3, 2026-05-26) "in production at leading AI labs" [FACT]; Meta **KernelEvolve** (ISCA 2026) generates Triton/CuTe/CUDA/HIP/MTIA kernels in prod [REPORTED]; Meta/PyTorch **KernelAgent** (2026-03-06): NCU-in-the-loop Profile→Diagnose→Prescribe loop, 1.56× geomean over torch.compile on KernelBench L1, 89% of roofline [FACT]. **Honest ceiling:** on FlashInfer-Bench (production kernels as baseline) best frontier models score **0.45–0.63×** — agents compile reliably, rarely beat SOTA; Standard Kernel's rubric: autonomous = solved for simple memory-bound ops, **human-guided CUDA+PTX still required to beat cuBLAS** (+6.0% geomean, H100). Your reps target exactly the tier AI can't do alone. |
| 9 | (not in note) | **Hardware roadmap update:** GB300 NVL72 shipping ("available now", 2026-02) — ~15 PF dense FP4, 288 GB, ~8 TB/s per GPU; Rubin VR200 in production, cloud H2-2026; **Rubin CPX quietly absent from GTC-2026 slides** (Groq-3 LPU, via the $20B NVIDIA-Groq license, now fills the decode-copro slot) [REPORTED, contested]; AMD MI355X GA (Oracle/Azure), OpenAI-AMD 6 GW + Meta-AMD 6 GW deals live, MI400/Helios 2H26-samples/Q2-27-volume; TPU v7 Ironwood GA (2026-04-22, 4614 TF/s FP8, 192 GiB @ 7.4 TB/s), Anthropic ≤1M-TPU deal; Trainium3 + **NKI stable** (Neuron 2.29/2.30). Multi-platform *reading fluency* is now a differentiator (§2). |

---

## 2. The 2026 hiring reality (what the roadmap must produce)

**Comp and demand [FACT where noted].** Anthropic posts **$280k–$850k** on both its GPU Performance
Engineer and TPU Kernel Engineer JDs [FACT, fetched]. Baseten GPU Kernel Engineer $185–250k base;
d-Matrix Staff SIMD $190–300k; OpenAI kernels ~$240–440k [REPORTED]. Inference-perf reqs outnumber
training-perf reqs at every company surveyed; kernel-titled openings run ~1.5–2 orders of magnitude
rarer than generalist-ML titles — thin supply, real premium. The "$1M kernel engineer" is an outlier
tail, not the market.

**The recurring JD nouns** (write these on the wall): CUDA/C++ · Triton · CUTLASS/CuTe · PTX/SASS ·
Nsight/NCU · roofline · tensor cores · FP8/FP4 · NCCL/NVLink · vLLM/SGLang internals · torch.compile ·
KV-cache/MoE-routing/attention kernels. Differentiators actually appearing in 2026 JDs: **AMD/ROCm-HIP**
(OpenAI runs a standing AMD-enablement req), **TPU Pallas / Trainium NKI reading fluency** (Anthropic
hires per-platform but screens cross-platform), and **"make AI-assisted optimization systems reliable"**
(OpenAI). xAI kernel MTS: GEMM via tensor cores "from scratch or CuTe/CUTLASS," register pressure,
Nsight [REPORTED].

**How you get read.** Anthropic's application form itself asks: *"link to, or describe in 1 paragraph,
the most impressive low-level or performance thing you've done"* [FACT]. Together AI's kernels-team
lead (Dan Fu) says on the record they hire people who "lose sleep over memory access patterns" and can
explain **why** it works, not that it works [FACT]. GPU MODE's co-founder publicly tells companies to
"hire the cracked engineers" straight off the public leaderboard [FACT]. The two demonstrated pathways:
**(1) a dated worklog with before/after numbers against a named ceiling; (2) upstream PRs into
vLLM/SGLang/FlashInfer/FLA/CUTLASS.** Both are extraction targets in §5.

**The interview loop shape (converged, 2025–26 reports):** live optimize-this-kernel or design-a-kernel
+ systems-design-for-inference (KV cache, batching, spec decode, quant) + C++/CUDA fundamentals +
roofline napkin math throughout + **"explain a kernel you wrote" deep-dive** — AI-prohibited, live.
This is exactly the mastery deficit's kill zone, hence §6.

**What this means for the roadmap:** every phase below must terminate in one of: a ledgered number vs
a named ceiling, an upstream PR, or a teach-back you can deliver cold. Anything else is inventory.

---

## 3. Vizuara curriculum — dissection and verdict

Six parts, 25 sessions, $3k, late-Sept 2026 start, lecture+live-coding pairs, capstone judged by a
"top SF GPU startup" (name-drops **Wafer** twice in its hiring section; Wafer = YC-backed, founded 2025
by Emilio Andere & Steven Arellano, $4M seed led by Fifty Years, angels Jeff Dean + Wojciech Zaremba,
product = autonomous AI GPU-performance-engineer agents [FACT]; sponsor identity [INFERENCE]).

**Session-by-session against this repo's ledger:**

| Vizuara session | Their endpoint | Your ledger (bench/RESULTS.md, notes) | Delta |
|---|---|---|---|
| I.0 CPU parallelism (SIMD/threads/OoO) | mental model | not built | **Steal as a 1-day awareness drill** (P0-Q6): AVX sum vs np.sum, why GPUs differ. Interview-adjacent, low priority |
| I.1 Roofline lab | predict-then-measure | A2 R0 harness; 0.55 TB/s / 72 TF/s peaks measured; constitution FOP-3 | Done, deeper than theirs |
| I.2 CUDA model + GPU puzzles | first kernels, read SASS | A2 R1–R4 GEMV→softmax→RMSNorm→TopK ladder, 96–100% HBM | Done; SASS-reading reps continue in P0 |
| I.3 Transpose ladder (coalescing/banks/occupancy) | Nsight lab | equivalent rungs measured; **ncu blocked** (ERR_NVGPUCTRPERM) | **ncu-debt is sm_120-bound — KVM 5090 ~$0.33/hr, NOT the H100 day (P1) it was misfiled to until 2026-08-30** |
| II.4–5 GEMM worklog (1.3%→36.5%→93.7% cuBLAS) | Simon Boehm arc (verified numbers: kernel-1 1.3% → warptiling 93.7%, A6000 FP32 [FACT]) | A2 R5/R6 GEMM 134% of cuBLAS-proxy on sm120; CUDA C++ smem GEMM built | Done (yours is tensor-core-era; theirs is FP32/SGEMM pedagogy) |
| II.6 WMMA tensor-core GEMM | WMMA beats SIMT | A3 R0–R2: 4.1%→38.9%→81.9% cuBLAS (mma.sync + wmma) | Done to 82%; the 94%+ tier is WGMMA/H100 = P1 |
| III.7 Profiling + debug 3 sabotaged kernels | pro NCU workflow | profiling discipline everywhere; **sabotage drills absent** | **Steal — best idea in their syllabus** → P0-Q4 standing drill |
| III.8 FlashAttention-1 build | FA1 fwd | A4 R0–R3 + bwd: FA2 fwd+bwd Triton, ~50% SDPA, 44× leaner | Done, beyond |
| IV.d1 FA1→FA2→FA3 | concepts | FA3 mechanism note + H100 runbook ready | P1 executes it |
| IV.d2 Beat cuBLAS on H100 (TMA/WGMMA/warp-spec) | >cuBLAS GEMM | A2 §4.1–4.6 runbook + compile-verified ISA kernels + WGMMA PTX artifact, **not yet run** | **= P1 rental day.** Reference: Pranjal's fast.cu (423 TF/s > cuBLAS) |
| IV.d3 Triton→CUTLASS→CuTe-DSL | modern authoring | Triton fluent; CuTe-DSL rung already re-pointed 07-09 | **= P2**; add TileLang awareness (FlashQLA/DSA) |
| IV.d4 Inference-serving kernels (paged/spec/quant) | concepts | A1 R0–R4.6 **all measured** (paged Triton kernel ×3.52, cudagraph 77%-of-wall, MLA toy, PD-disagg demo, n-gram spec decode); A5 quant ladder incl. NVFP4 1.48×<MXFP4, FP8-KV 24.45 dB | Done, far beyond — theirs is survey-level |
| IV.d5 Blackwell tcgen05/TMEM/NVFP4 | concepts + demo | B200 runbook written; sm120 lacks TMEM/tcgen05 [FACT] | **= P2 rental day** (CUTLASS 4.5 + TK-2.0 as references) |
| IV.d6 FA4 deep-dive | "understand before almost anyone" | §1 row 1 — you now hold the verified technique list | P2 study + bench vs cuDNN 9.24; then teach-back |
| V.d7 DeepSeek FlashMLA/DeepGEMM | reading | MLA toy built; DeepGEMM = A6/wide-EP reading | P3 uses DeepGEMM-style JIT patterns |
| V.d8 LLM-generated kernels (KernelBench/AlphaEvolve, agent+profiler loop) | survey + guided lab | **absent from your plan** | **= new module M-AI** (§4) — cheap, JD-relevant |
| VI Capstone (real startup problem, judged) | demo day | DELTA is harder and yours; judges = the open market (FLA maintainers, GPU MODE, hiring managers) | P3/P4 |
| "Worklog becomes a book" | illustrated book | A1–A7 design notes exist = 7 chapters already drafted | **Steal the packaging** → P4 public serialization |

**What the workshop genuinely gets right:** frontier-current topics (FA4/Blackwell/NVFP4/AI-kernels);
lecture↔live-coding pairing; "why it matters / who uses it" framing; a graded capstone; the
worklog-as-artifact instinct. It is likely the best *taught* kernel program on the market right now.

**What it omits that production reality (and your target JDs) demand:** backward-pass kernels & training-side
perf (their arc is inference-only; your FA2 bwd + rental FA3 bwd cover it) · MoE **grouped-GEMM** and
EP comm kernels (DeepEP/EPLB/DBO — your A6) · communication/overlap kernels (NCCL device API, symmetric
memory, async-TP) · determinism & batch-invariance (Thinking Machines 2025-09; now a vLLM/SGLang product
feature and an RL-correctness lever — your A5/RL twin) · quantization *accuracy* discipline (SQNR,
downstream evals — your A5) · perf-regression CI & kernel packaging (torch custom ops, torch.compile
interop) · multi-platform hedge (ROCm/HIP, Pallas/NKI reading) · and **reps under time pressure without
an instructor** — the thing interviews actually test.

**Marketing-claim fact-check (calibration on "trust but verify"):** "FA4 = first attention kernel to
break a petaflop" — **false as stated**: FA3 FP8 hit ~1.2 PF on H100 in 2024 [FACT]; FA4's 1.6 PF is
the BF16 Blackwell landmark. "Folded into cuDNN" — techniques adopted, not a merge. "Written by AI"
framing (their Part V) is accurate as a *survey* but the honest mid-2026 ceiling is 0.45–0.63× of
production kernels (§1 row 8). Their GEMM numbers (1.3%→93.7%) are Simon Boehm's real, verified arc.

**Verdict (V2 restated):** for a beginner, worth it; for this repo's owner, **strictly dominated** by
executing your own rental runbooks + M-AI + publishing. Skip; waitlist optional; poach the sabotage
drill, the CPU session, and the book packaging.

---

## 4. The roadmap — phases, rungs, DoD

> Delta-based: extends `performance/A1–A7` + `GPU_FROM_ZERO` rungs 0–9; nothing already ledgered is
> repeated. Every rung keeps the constitution: pre-registered prediction → build → **profile as DoD** →
> ledger entry → teach-back. Costs are [INFERENCE] estimates; runbooks already exist for P1/P2.
> **Gate: P1+ opens only per MASTER_PLAN (post-flagship-curve, Week 10+ for DELTA).** P0 and M-AI-lite
> run inside the 15% background allocation.

### P0 — Quarry (daily block-opener; 30–45 min; $0; retrieval only — runs for the life of the lane)
The mastery protocol of §6 applied to **already-built** kernels. Output = teach-backs ledgered against
MASTERY_DEBT, not code. Rungs (Q1–Q6) are defined in §6. *Exit criterion: the 10-item "defend cold"
surface (§7) at teach-back grade ≥ B.*

### P0.5 — The Rebuild Ladder (mastery-first; ≈ Jul 15 → Aug 1; the deep-work slot until done)

> **Why (added 2026-07-15 at the user's direction).** The production kernels were agent-built under
> delegate mode — the ledger is real, the human's generative skill is not yet (4/89 teach-backs).
> Interviews are AI-prohibited; memory forms only from what the human generates (§6.5 evidence). So
> before any rental day: re-derive and re-type the core ladder from blank in `mastery/src/mastery_llm/`
> (the 100%-mirror skeleton + rewired tests that already exist), via the **`/rebuild`** command's
> 6-phase loop: derive on paper → timed worked-example study of the agent version (then it closes) →
> blank rebuild vs the existing oracle → predict-then-bench vs the ledgered number → **diff-defend**
> every divergence against the agent's version → `/feynman` + spaced re-queue (+48h/+1wk).
> Rungs L0–L8 (exec model · roofline · GEMV · softmax · RMSNorm · tiled GEMM · online-softmax→FA2
> fwd+bwd-derivation · decode/paged step · quant numerics) ≈ 10.5 blocks. **Consequence, accepted:**
> P1/P2 rental days slide ~2 weeks — measuring agent-written WGMMA kernels before the human owns the
> sm120 ladder would mint more indefensible inventory. `mastery/src/**` is human-only; agents never
> write there. M-AI slides to background/September (it is delegate-friendly and benefits from
> mastery-first anyway).

### P1 — Hopper day (1× H100 rental, ~6–10 h, ≈ $25–60; runbooks: `performance/rental/`)
| Rung | Build/measure | Named ceiling | Pre-registered prediction (write exact numbers in the runbook before boarding) |
|---|---|---|---|
| P1.1 | ncu-debt list from A1–A5 (profiler unblocked on rental) | — | each kernel's SoL bucket matches the sm120 analytic attribution |
| P1.2 | WGMMA+TMA persistent GEMM (A2 §4.1–4.6, compile-verified) | cuBLAS on locked clocks | ≥80% cuBLAS; without WGMMA caps ~63% — measure both |
| P1.3 | warp-specialized pipeline variant (pingpong) | own P1.2 | the async overlap closes ≥half the gap to cuBLAS |
| P1.4 | FA3-style fwd (TMA+WGMMA (+FP8 stretch)) — A4 R4 | FA3 (~740 TF/s BF16 / ~75% util [FACT]) + cuDNN | state your % target *before*; diagnose the gap per-component |
| P1.5 | (stretch) FP8 GEMM w/ block scaling | DeepGEMM (~1350 TF/s-class on Hopper) | accumulation-trap demo: two-level accumulation recovers accuracy |

### P2 — Blackwell day (1× B200 rental, ~6–10 h, ≈ $50–120)
| Rung | Build/measure | Named ceiling | Note |
|---|---|---|---|
| P2.1 | tcgen05/TMEM GEMM **via CUTLASS 4.6 / CuTe-DSL** (use, extend, profile — don't reimplement) | cuBLAS + TK-2.0's published B200 GEMMs | the 07-09 "vendor-served → understand+use" ruling stands |
| P2.2 | **CuTe-DSL authoring rung**: port one owned kernel (RMSNorm or GEMV) to CuTe-DSL; read the generated PTX/SASS | your own Triton version | this is the FA4/PyTorch-2.13-backend dialect — the JD-visible skill |
| P2.3 | NVFP4 block-scaled GEMM (A5 §7) | FP8 baseline | speedup **and** SQNR + downstream-task delta; NVFP4(16-blk, E4M3 scale) vs MXFP4(32-blk, E8M0) mechanism [FACT] |
| P2.4 | FA4 study-bench: run `flash-attn-4` vs cuDNN 9.24 vs FA3-port; read `flash_attn/cute/flash_fwd_sm100.py` | FA4's ≈1.6 PF/71% | teach-back the four tricks (§1 row 1) + the Hopper-decode regression |
| P2.5 | (stretch) TileLang hello-kernel (FlashQLA's dialect) | — | 1–2 h awareness, not mastery |

### P3 — DELTA (from ≈ Aug 12 per the 07-15 rebuild-first resequence; 2–4 wk; the capstone; spec in `../DELTA.md` + PERF_PLAN §6 re-scope)
The **FP8 (→NVFP4-state stretch) fused recurrent GDN decode-step kernel**, married to the F10
model-side Gated-DeltaNet: (a) oracle = FLA `fused_recurrent_gated_delta_rule` bit-accurate reference +
quality eval on F10 checkpoints; (b) baseline to beat = FLA BF16 fused_recurrent at matched quality;
(c) targets: ≥85% of memory roofline on the dev box first, then the B200 number; (d) pre-registered
kill criteria per the DELTA P1–P7 shape; (e) **the FlashInfer MLSys-26 GDN-track harness and FlashQLA
give you ready-made benchmark scaffolding** [FACT]. Ship = §5 extraction (FLA PR *or* standalone repo)
+ worklog chapter. This is the Anthropic-application-question answer.

### P4 — Production integration & photons (parallel with applications; 1–2 wk spread)
| Rung | Artifact | Why (verified) |
|---|---|---|
| P4.1 | **One upstream PR** to FLA / vLLM / SGLang / FlashInfer (a kernel, a fix from your ncu-debt findings, or DELTA itself) | pathway #2 in every hiring stream; engagement with others' code is the screened trait |
| P4.2 | Package one kernel as a **torch custom op** (`torch.library`, compile-compatible, tests + microbench in CI) | "production kernel" ≠ fast kernel; it's versioned, tested, regression-gated |
| P4.3 | Serialize the worklog: A1–A7 notes + rental-day posts + DELTA writeup → public site/blog series | "private work emits no photons"; Anthropic's form asks for the link [FACT] |
| P4.4 | GPU MODE leaderboard submission when the next comp opens (they run ~quarterly: AMD $100k '25, NVFP4 hackathon '25-26, AMD E2E speedrun '26) | the co-founder-stated hiring funnel [FACT] |

### M-AI — AI-written-kernels module (2–3 days, standing box, can run pre-Week-10 as background)
Build a **minimal NCU/profiler-in-the-loop agent harness** (the KernelAgent pattern: Profile → Diagnose
→ Prescribe → Measure) over 3–5 KernelBench L1/L2 tasks + one of your own kernels; write the honest
eval note: where it beat torch.compile, where it hallucinated, how you'd verify against gaming
(fuzzed shapes + fp64 oracle — the robust-kbench lesson; two documented reward-hacks in '25–'26 comps).
*Why:* it is now a JD line-item (§2), the Wafer/Standard-Kernel thesis, and you already live in agent
tooling — this converts daily practice into a named, defensible artifact. **You supervise; the point
is the eval discipline, not the agent.**

### M-AMD — optional hedge (defer unless targeting an AMD-enablement req)
Reading tier now (HipKittens paper 2511.08083; AITER as ROCm's default kernel lib; Triton-on-ROCm
maturing): 2–3 h. Build tier only if a target req demands it: rent MI300X/MI355X for a day, port ONE
Triton kernel + profile with rocprof. OpenAI-AMD 6 GW + Meta-AMD 6 GW make this premium-but-optional
[FACT]. Same logic for Pallas (TPU) / NKI (Trainium): **reading fluency for Anthropic loops** — 2 h
each with their docs; NKI went stable in Neuron 2.29 (2026-04).

---

## 5. Embed vs separate — the mechanics (V1 expanded)

**Three tiers, one rule: learn embedded, publish extracted, contribute upstream.**

| Tier | Lives where | What |
|---|---|---|
| **Learning + integration** | `scratch_llm/performance` (unchanged) | curriculum, rungs, benches, oracle tests, ledger, runbooks. The serving engine, F10 model, RL loop are the *integration targets* that make kernels non-toy — this co-location is the moat; do not fork it |
| **Portfolio exits** | extracted **when the number exists** | DELTA → **prefer an FLA PR** (instant distribution + review by Songlin Yang's lineage; fallback: standalone `gdn-fp8-decode` repo with pip bench harness, DeepGEMM's ~300-line aesthetic); rental-day worklogs + FA4 note → public blog series; M-AI harness + eval note → small public repo |
| **Real-world problem solving** | other people's repos | the P4.1 upstream PR; GPU MODE comps; (optional) FlashInfer-Bench/KernelBench contributions |

Extraction is a ≤1-day packaging task per artifact — never a new "project" with its own planning
surface. Repo hygiene: this file + ledger entries are the kernel lane's entire doc footprint.
**Context wiring (landed 2026-07-14, user-requested):** `CLAUDE.md` perf-front bullet (always-on, one
line) · `PERF_PLAN.md` 07-14 orientation block w/ maintenance contract (on-demand) · `PERFORMANCE_TRACK.md`
header refresh pointer · `JOB_SPRINT/MASTER_PLAN_2026-07-12.md` §3.2 kernel-lane line (cross-repo
parallelism with the flagship) · `.claude/commands/kquarry.md` (the daily P0 rep, learn-mode-always).
Update protocol: phase boundaries → `PERF_PLAN` Current Node + §7 milestones here + re-verify §1/niche;
daily reps → mastery ledger only; everything else is code, benches, or postmortems.

**Why not a separate learning repo (explicit):** (1) your kernels already plug into your own engine —
that *is* the production shape interviewers probe; (2) infra reuse (hooks, oracles, ledger, CI); (3) a
separate repo restarts the doc-to-code spiral MASTER_PLAN just killed; (4) hiring evidence rewards
upstream PRs and worklogs, not new personal frameworks. **Why not fully private:** photons — the
Anthropic form, the leaderboard funnel, and the worklog pathway all require public surface; extraction
solves it without moving the curriculum.

---

## 6. The learning protocol — how we run this together (V4 expanded)

**Mode contract.** Kernel-lane sessions run `execution-mode = learn` (ADR-0013's historical contract,
enforced by `kernel-write-guard.sh`): **you type every kernel body and every derivation; the AI is
forbidden to.** The AI's five legitimate jobs: (1) **Socratic tutor** (`/tutor`, `/master`) — derive
via questions, never hand the derivation; (2) **scaffolder** — failing oracle tests, bench harnesses,
shape/dtype adversarial cases; (3) **profiler analyst** — read nsys/ncu output *with* you, name the
next unmodeled constraint (launch/latch/occupancy/bank-conflict — FOP-3); (4) **saboteur** — break
kernels for your drills (below); (5) **adversarial reviewer** (`/kreview`) — post-hoc, against the
frontier (e.g., "TK-2.0 does this with one fence — why do you have three?"). Exception: M-AI runs in
delegate mode — there the *agent harness* is the artifact and your job is supervision + eval.

**The P0 quarry reps (pick 1/day, 30–45 min, ledger every rep):**

| Rep | Drill | Gate |
|---|---|---|
| **Q1 Blank-page ladder** | rewrite from memory, timed, no AI: online-softmax (20') → naive+tiled GEMM (30') → FA2 fwd inner loop (45') → fused paged-decode step (45') → GEMV ladder (30'). Cycle weekly | compiles + passes the existing oracle test; else re-derive tomorrow |
| **Q2 Predict-the-number** | AI picks a random ledgered kernel from `bench/RESULTS.md`; you re-derive its bytes/FLOPs, bound, and % of peak cold; then compare to the measured row | within ±20% + correct bound classification |
| **Q3 Teach-back** | 10-min whiteboard-style explanation of one MASTERY_DEBT row; AI quizzes with one modify-and-predict variation | AI grades A–F vs the note; ≥B to clear the row |
| **Q4 Sabotage drill** *(stolen from Vizuara, upgraded)* | AI plants 1–3 realistic bugs in a copy of your kernel (race, missing `__syncthreads`, bank conflict, wrong swizzle, off-by-one mask, silent dtype cast, fence removal); you get the failing/slow symptom only; fix under 25' with profiler | root-cause named *before* the fix; postmortem sentence ledgered |
| **Q5 PTX/SASS rep** | `nvcc -ptx` / godbolt one owned kernel; annotate 10 lines; find one thing the compiler did for/against you | one concrete finding ledgered |
| **Q6 Frontier read** | one primary source/week from §8 (TK-2.0 post, FA4 paper §3, DeepGEMM README, NCCL 2.28 blog, vLLM large-scale-serving) + 5-line summary tying it to a rung | summary survives AI cross-examination |

**Cadence (dual-block, re-amended 2026-07-14 evening).** Daily: the quarry rep (30–45', unchanged
gates) **opens the 14:00–18:30 kernel block**; deep work follows per the §4/§7 progression. The 06:30
morning rep + block belong to the RL lane; the 21:00 debrief grades both lanes (quarry gate + the
block's DoD). Slipped flagship day → deep work yields, quarry Q2/Q3 short-form survives. Weekly: one
Q6 read + MASTERY_LEDGER review; kill or promote drills like ablations. Rental days and
P3 use the full 6-step forced-mastery loop from `GPU_FROM_ZERO` (derive → visualize → predict → build
test-first → teach-back gate → connect-to-frontier) — unchanged, it is already correct.

**Interview-surface additions** (append to GPU_FROM_ZERO's "defend cold" list): tcgen05/TMEM/CTA-pair
mental model + sm100-vs-sm120 split · NVFP4 vs MXFP4 (block 16/32, E4M3/E8M0 scales, why NVFP4 is
finer) · FA4's four tricks + why softmax became the bottleneck (asymmetric hardware scaling) + the
Hopper-decode regression · wide-EP anatomy (DeepEP normal/LL-IBGDA, EPLB, DBO, PD-disagg econ:
DeepSeek's 545%-margin report) · NCCL device API three modes (LSA/Multimem/GIN) & why comm moved
on-device · batch-invariance root cause (batch-size-dependent reduction order) and its RL train↔infer
consequence · spec-decode acceptance math (EAGLE-3 ~0.75–0.85, MTP, DSpark) · "how would you evaluate
an AI-generated kernel?" (fuzzed shapes, fp64 oracle, reward-hack war stories) — this last one is the
new 2026 curveball [INFERENCE].

---

### §6.5 The Claude stack — evidence-calibrated collaboration (researched 2026-07-14)

> Two primary-source research streams (Claude Code/Cowork lab practice · learning-science 2025–26).
> This section sets HOW the AI is used in every kernel-block hour: what it may do, what it must never
> do, and why — each rule tied to a dated finding.

**The evidence → the rules:**

| Finding (primary, dated) | Rule it sets in this lane |
|---|---|
| **Anthropic skill-formation RCT** (arXiv 2601.20245, 2026-01-29): AI-assisted learners scored **50% vs 67%** (d=0.738) on immediate quizzes, worst gap on **debugging**; but interaction pattern decided everything — *conceptual-inquiry-only* and *generation-then-comprehension* scored ≥65%, *full delegation / progressive reliance* <40% | Mode-3 typing is the production function (constitution confirmed by RCT). During reps the AI answers **concept questions only**. Debugging is the most protected skill: Q4 stays manual-first, AI debug loops capped at 10 iterations (AWS NKI discipline) and you state your hypothesis **before** asking |
| **METR RCT** (2025-07-10): experienced devs **19% slower** with AI while believing +20%; 2026-02 follow-up: −18%/−4%, CIs cross zero | Delegation is not free speed. Delegate only when **verification ≪ generation** (quadrant below). Never trust felt speedup — the calibration ledger measures it |
| **Anthropic AI Fluency Index** (2026-02): "the better the output looks, the less people question it" | **Interrogation pass**: before accepting any AI explanation/summary, name one flaw, missing caveat, or boundary condition. Wired into Q6 and `/master` |
| Worked-example + expertise-reversal literature | **Attempt-first ordering** (kernel-tutor's human-first rule is correct — keep). Full worked examples only on first contact with genuinely novel mechanisms (tcgen05, first CuTe layout), fading after |
| Retrieval-practice study (arXiv 2507.05629): AI-quizzing lifted accuracy **73%→89%** | **AI quizzes > AI lectures.** Q2/Q3/`/feynman` are the default mode; `/master`'s teaching mode is for first contact only |
| MIT cognitive-debt EEG (2506.08872, small-n caveat) + Anthropic's "paradox of supervision" (2025-12-02: engineers deliberately work manual "to keep myself sharp"); NVIDIA/HRT explicitly ban AI in live rounds | The no-AI quarry reps are non-negotiable — train the exam condition |

**Delegation quadrant (the save-time half).** DELEGATE freely (cheap to verify): test scaffolds,
bench harnesses, plot/log parsing, sweep runners, runbook checklists, viz artifacts, doc chores —
verified by green tests / visible plots / diffs. DELEGATE with caps: refactors ≤400 lines (the
rubber-stamp cliff — Finster 2026-03). NEVER delegate (high stakes × hard verify): kernel bodies and
derivations you'll defend live · numerics decisions · first-pass profile interpretation (AWS NKI:
interpreting the profile **is** the human job) · pre-registered predictions.

**Claude Code mechanics for this lane** (Anthropic living best-practices + Cherny/practitioner-verified):
plan mode before any multi-file change (reported 2–3× success rate) · always hand Claude a verifier —
our oracle test + bench IS the verifier (FOP-3) · **Learning output style** for scaffold work in
kernel-block sessions: `/config` → Output style → **Learning** — Claude writes the harness and leaves
`TODO(human)` markers at the strategic lines (verified alive in docs 2026-07-14; the `/output-style`
command was removed 2026-04, the styles were not) — the middle ground between delegate and learn
modes; `kernel-write-guard` still owns the hard boundary · `/clear` per rung; PERF_PLAN "Current
node" is the handoff file (our standing pattern — now the externally validated HANDOFF.md idiom) ·
rental days: parallel agents/worktrees for **harness work only** (one agent preps the next rung's
bench while you write the current kernel) · subagents return summaries, not dumps (context rot).

**The five roles → machinery** (✓ exists · ★ new today): DERIVE ✓ `/tutor`, `/master` 1–3 ·
VISUALIZE ★ `/kviz` + use-first list below · PREDICT ✓ Q2 + FOP-3 · BUILD ✓ learn-mode + ★ Learning
style for scaffolds · PROFILE ✓ `/profile` + roofline-analyst (+ the 10-iteration cap) · REVIEW ✓
`/kreview` · TEACH ★ `/feynman` (+ phone voice-mode viva, 30–45 s utterance chunks) · READ ✓ Q6 +
citation-tree (+ interrogation pass).

**Block choreography (14:00–18:30, non-rental days):** 14:00 quarry rep (30–45') → 14:45 DERIVE 10'
(Socratic, attempt-first) → VISUALIZE 10' (`/kviz`: open or build; predict each step before advancing)
→ PREDICT 5' (numbers written down) → BUILD ≈2.5 h (learn mode / Learning style; Claude scaffolds
tests + bench in parallel) → PROFILE 20' (you read the profile first, then the analyst; name the next
unmodeled constraint) → `/feynman` 15' → ledger 5'. Rental days: the runbook replaces this.

**The viz library — `performance/viz/` (use first, build only on verified gaps):**
USE: Triton-Viz (SIGCSE'26-validated pedagogical visualizer) · Compiler Explorer CUDA (PTX/SASS,
source-correlated) · Modal GPU Glossary · Nsight Compute's built-in roofline · `cute::print_layout`
/ `print_latex` · Boehm + Gordić worklogs · GPU MODE lecture notes (C. Mills) · Mojo GPU Puzzles'
roofline explorer. BUILD (gaps verified 2026-07-14; single-file interactive HTML, tiny numeric
example, one per mechanism): **#1 online-softmax → FA4 lazy-rescale stepper (seeded ✅ 2026-07-14)** ·
#2 CuTe layout/swizzle explorer (no public interactive tool exists) · #3 paged-KV block-table
animator · #4 GDN/delta-rule recurrence stepper (DELTA prep) · #5 coalescing/bank-conflict grid.
Rule: a viz you watched passively is decoration — predict each step, then `/feynman` the mechanism.
Building viz code = Mode-1 delegate (teaching aid, not the tested skill).

## 7. Milestones → hire-ready (dated, subordinate to MASTER_PLAN)

| When | Milestone | Evidence produced |
|---|---|---|
| Jul 15 → ≈ Aug 1 (dual-block) | **P0.5 Rebuild Ladder L0–L8** (`/rebuild`, mastery/, from blank) + P0 quarry daily | every core kernel re-typed by the human vs oracle + ledger; teach-backs 4/89 → ≥30/89; diff-defend notes |
| ≈ Aug 1–2 (post-ladder) | **P1 H100 day** — now the human hand-writes the WGMMA upgrade from THEIR L5 GEMM (agent version = reference-after-attempt) | ≥80%-cuBLAS WGMMA GEMM + FA3-sight number — numbers the human can defend cold. **(30/08: "ncu-debt cleared" REMOVED from this DoD — 4 of 5 metrics are sm_120 and clear on the $0.33/hr KVM, not here; only WGMMA tensor-pipe util belongs to this day.)** |
| ≈ Aug 6–8 | **P2 B200 day** | tcgen05/CuTe-DSL/NVFP4/FA4 ledger rows + worklog post |
| from ≈ Aug 12 (niche re-check at open) | **P3 DELTA** — human-typed by construction; the ladder was its training camp | the niche kernel vs FLA baseline, roofline-placed, quality-evaluated |
| Sep, parallel with applications | **P4 extraction volley** | FLA/vLLM PR merged-or-in-review · public worklog series · torch-custom-op CI · comp submission if open |
| application answers | Anthropic form: *"FP8 recurrent-state GDN decode kernel, X% of B200 memory roofline, beats FLA BF16 baseline at matched quality — co-designed with the hybrid model it serves; worklog: <link>"* | one paragraph, one link, exactly what the form asks |

Target tiering per the hiring stream: **Tier 1** (niche-aligned): Thinking Machines (Songlin Yang's
team - GDN lineage), Together AI kernels (Dan Fu/Tri Dao), Anthropic perf (GPU or TPU track) ·
**Tier 2**: OpenAI inference-kernels/AMD-enablement, xAI kernel MTS, NVIDIA CUTLASS/TRT-LLM/DevTech ·
**Tier 3** (velocity offers): Baseten, Fireworks, RadixArk (SGLang's company), Modal, Wafer,
Standard Kernel, Mako. Budget total: **≈ $100–400 of rentals + $0 tuition.**

---

## 8. Sources (dated primaries; fetched or cross-corroborated 2026-07-14)

**Kernels/attention:** arXiv 2603.05451 (FA4, 2026-03-05) · tridao.me/blog/2026/flash4 ·
modal.com/blog/reverse-engineer-flash-attention-4 (2025-09-26) · arXiv 2407.08608 (FA3) ·
hazyresearch.stanford.edu TK-2.0 (2026-02-19) · arXiv 2511.08083 (HipKittens) · siboehm.com CUDA-MMM ·
cudaforfun.substack.com H100 worklog (2024-11-29) · github.com/QwenLM/FlashQLA (~2026-04-29) ·
github.com/fla-org/flash-linear-attention · arXiv 2510.26692 (Kimi Linear/KDA) · arXiv 2603.15569
(Mamba-3, ICLR26) · arXiv 2502.11089 (NSA).
**Vendor/hardware:** CUDA 13.3 release notes · github.com/NVIDIA/cutlass releases (4.5.2, 2026-06-16) ·
pypi nvidia-cutlass-dsl · NVIDIA NCCL 2.28 device-API blog · nvidia.com GB300 NVL72 page (2026-02) ·
Tom's Hardware Rubin-CPX-removed (2026-03-17) · docs.cloud.google.com/tpu/tpu7x (Ironwood, 2026-06-18) ·
AWS Neuron 2.29/2.30 (NKI stable) · amd.com OpenAI-6GW (2025-10-06) & Meta-6GW (2026-02-24) · ROCm 7 /
AITER · NVIDIA CompileIQ blog (2026-05-26).
**Serving/distributed:** vLLM releases v0.25.0 (2026-07-11) + large-scale-serving blog (2025-12-17) +
FP8-KV postmortem (2026-04-22) + DeepSeek-V4 day-0 (2026-04-24) · lmsys.org large-scale-EP (2025-05-05)
+ deterministic-inference (2025-09-22) · NVIDIA Dynamo 1.0 (GTC 2026) · github.com/deepseek-ai/
{DeepGEMM, FlashMLA, DeepEP, EPLB} · arXiv 2606.19348 (DeepSeek-V4) · DSpark (2026-06-27, arXiv
2607.05147) · pytorch.org 2.13 blog (2026-07-08) + MXFP8+DeepEP TorchTitan post · thinkingmachines.ai
defeating-nondeterminism (2025-09).
**AI-kernels/DSLs:** pytorch.org KernelAgent (2026-03-06) · engineering.fb.com KernelEvolve (2026-04-02)
· standardkernel.com seed/rubric (2026-03-11) · deepmind AlphaEvolve (2025-05-14: 23% Pallas-kernel →
1% train-time; 32.5% FA) · Sakana retraction (2025-02) · metr.org kernel eval (2025-02-14) ·
arXiv 2502.10517 (KernelBench) · robust-kbench (2509.14279) · FlashInfer-Bench (2601.00227) + MLSys-26
contest (mlsys26.flashinfer.ai) · gpumode.com leaderboard/comps · huggingface.co/blog/upskill (2026-01-28).
**Hiring:** greenhouse Anthropic JDs 4720576008/4926227008 ($280–850k, fetched) · openai.com careers
(inference-CUDA-kernels; AMD-enablement; kernel-perf-AI-tooling) · together.ai kernels-team post
(2026-04-01, Dan Fu quotes) · x.com/marksaroufim leaderboard-hiring post · ycombinator.com/companies/wafer ·
baseten/d-Matrix/xAI/Cerebras postings · levels.fyi bands.

*Known source conflicts, resolved:* TK-2.0 date 2026-02-19 (primary fetch) over "Jan 11" (secondary) ·
FA4 1605 (blog) vs 1613 (abstract) TF/s → cite ≈1.6 PF · V4-Flash 284B (official) not 285B · Rubin CPX
status contested (absent from GTC-26 slides; no official cancellation) · Cursor's serving stack:
Together/TRT-LLM+NVFP4 per Together's case study, SGLang claims conflicting [UNVERIFIED].
