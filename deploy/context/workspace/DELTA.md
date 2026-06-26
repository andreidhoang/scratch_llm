# DELTA — GatedDeltaNet-2 decode kernel (capstone)
### A fused **decode-step** kernel for GatedDeltaNet-2: making erase/write decoupling free at the memory-bound roofline

> **Merged doc.** Part I is the design RFC; Part II is the 4-week execution plan. It supersedes and replaces the two original working docs (the capstone design doc + the execution plan), now removed — this is their canonical merge.

> **Landscape context (reference):** the broader 2026 serving-stack / inference-frontier this decode-kernel spike sits inside — frameworks, prefill/decode disaggregation, quantization (NVFP4), the frontier-model table, hiring signals — is mapped in [`reference/Frontier_Inference_2026_Research_Brief.md`](reference/Frontier_Inference_2026_Research_Brief.md) (+ a compute-tiered [`curriculum`](reference/Frontier_Inference_Engineering_Curriculum_2026.md) alongside it). Reference only; **DELTA remains the canonical capstone spec.**

**Status:** verified by 5-lens design panel · fact-checked against primary sources 2026-06-14 (arXiv 2605.22791 + NVlabs repo confirm the architecture; **key correction: NVlabs released training code only — no checkpoint — so e2e is rescoped to projected/proxy**) · **Target artifact:** public repo + design doc + postmortem + benchmark report + bilingual writeup · **Sequencing (as of 2026-06-20): base-first** — the kernel build is gated behind the **A5 RL ship + Step-0** (see [`STRATEGY.md`](STRATEGY.md) §8); the weeks in Part II are *relative sprint weeks*, not calendar dates
**Author:** Huy · **Reviewers (in-context panel):** kernel/perf · linear-attn architecture · eval/methodology · inference-systems · adversarial verifier

> Format follows the frontier Phase-2 spec discipline: one-sentence hypothesis → falsifiable numeric predictions → scope → baselines → eval harness → kernel design → ablation plan → kill criteria → reproducibility → signal map. Epistemic labels: **[FACT]** grounded in cited 2026 sources · **[INFERENCE]** reasoned conclusion · **[UNCERTAIN]** to be measured.

---

## 1. Context and the gap (why this is the right artifact)

**[FACT]** GatedDeltaNet (GDN) is the workhorse linear-attention layer of the current hybrid-model wave. Qwen3-Next, Qwen3.5/3.6, Qwen3-Coder-Next, and Kimi Linear all interleave linear-attention and full/sliding-window-attention layers at roughly a **3:1 ratio** (≈three GDN blocks per non-linear block). *(Raschka, LLMs-from-scratch ch04/08; Qwen.)*

**[FACT]** GatedDeltaNet-2 (NVIDIA — Hatamizadeh, Choi, Kautz; arXiv 2605.22791, submitted 21 May 2026) generalizes GDN and KDA by **decoupling the active memory edit** into a channel-wise **erase gate `b_t ∈ [0,1]^{d_k}` on the key axis** and a channel-wise **write gate `w_t ∈ [0,1]^{d_v}` on the value axis**, while preserving channel-wise decay `D_t = Diag(α_t)`. The reference update is `S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t−1} + k_t (w_t ⊙ v_t)ᵀ`. It recovers KDA when both gates collapse to a scalar `β_t`, and GDN when the decay collapses too. It admits a chunkwise WY form and ships **fused Triton kernels for training only** (WY chunkwise + gate-aware backward). **Critically for this capstone: no decode-step kernel and no pretrained checkpoint are released** — the NVlabs repo is *train-your-own* code under a non-commercial (NVIDIA Source-Code-NC) license. Released config: **16 heads, `d_k = d_v = 128`**, so state `S ∈ ℝ^{128×128}` per head. *(arXiv 2605.22791; github.com/NVlabs/GatedDeltaNet-2.)*

**[FACT]** Training/prefill kernels are **already optimized and owned**: `fla` (flash-linear-attention, Songlin Yang) is the canonical Triton baseline — and is an explicit GDN-2 dependency (listed in the NVlabs acknowledgements). **[UNCERTAIN]** Qwen's **FlashQLA** (TileLang) *reportedly* delivers ~2–3× forward / ~2× backward over the `fla` Triton kernel for GDN chunked prefill on Hopper — **verify the FlashQLA source before citing these multipliers as fact** (not independently confirmed here). Either way the load-bearing point holds: **prefill/training are owned; decode is the open gap.** *(fla GitHub; qwen.ai/blog FlashQLA — to verify.)*

**[FACT]** Decode is the open, bandwidth-bound regime. At batch-1, **GDN decode is memory-bound because the full recurrent state is round-tripped through HBM every token**; all subquadratic models (GDN, DeltaNet, Mamba, Mamba-2) sit **below 1 FLOP/B** on the H100 roofline at decode — *more* memory-bound than standard attention. *(USC, arXiv 2603.05931.)*

**[INFERENCE]** The naive decode path is catastrophic: the GDN recurrent step decomposes into ~20 elementwise/matmul ops that **read and write the state to global memory ~5 times per token** (decay → retrieve → delta → update → read); a fused kernel that keeps the state register/SMEM-resident closes a large bandwidth gap, and at 75% GDN layers this overhead dominates end-to-end inference. State traffic is **config-dependent**: for the released 1.3B config (16 heads, `d_k=d_v=128`, bf16) the per-head state is **32 KB** (128·128·2 B) → **512 KB across the 16-head layer** per token — pin this to the exact SKU/dtype you measure on. *(HBM-pass mechanics are independently sound; the specific **ONNX issue #7689 / "10–50×"** figure is **[UNCERTAIN]** — not verified here.)*

**[INFERENCE] The gap this capstone fills:** a fused **decode-step** kernel for **GatedDeltaNet-2 specifically** (erase/write decoupled), on **GPU** (the USC restructuring was FPGA + plain GDN), that reduces state HBM traffic from ~5 passes to ~2 (one read, one write) via algebraic restructuring, and demonstrates that **the erase/write decoupling is nearly free at decode** because the operation is bandwidth-bound on `S`, not on the gate vectors. This is non-redundant with NVIDIA's training kernels, FlashQLA's prefill kernels, and the USC FPGA work — three-way clean differentiation, confirmed against the NVlabs repo (training-only kernels, no decode path, no released checkpoint) and the USC paper title (explicitly *on FPGA*).

---

## 2. Hypothesis (one sentence)

> Fusing the GatedDeltaNet-2 decode recurrence into a single state-resident kernel that touches each per-head state matrix with **one read pass and one write pass per token** will reach **≥85% of the H100 memory-bound roofline** for the state update, and will make **GDN-2 decode latency within ~10% of plain-GDN decode latency** despite the two additional channel-wise gates — because decode is bandwidth-bound on `S`, so the decoupling's extra arithmetic is amortized.

---

## 3. Falsifiable numeric predictions (write before building)

| # | Prediction | Falsifier (abandon/investigate if…) |
|---|---|---|
| P1 | **Roofline:** fused state-resident kernel achieves **≥85%** of the **measured** achieved-DRAM-bandwidth roofline at batch-1 (Nsight Compute `dram__throughput.avg.pct_of_peak_sustained_elapsed`). Denominator = empirical bandwidth from a STREAM/empty-copy microbench on the **exact** H100 SKU (SXM vs PCIe, HBM3 vs HBM2e swing it ~25–30%), **not** theoretical peak | <70% → state is not actually resident; a pass is leaking to HBM |
| P2 | **Free-decoupling thesis:** GDN-2 decode latency ≤ **1.10×** plain-GDN decode latency at batch-1 | >1.25× → the extra gate loads are not amortized; kernel is leaving bandwidth on the table |
| P3 | **vs `fla` recurrent:** match or beat the `fla` recurrent GDN kernel; reach **≥90%** of its tok/s on the GDN case before claiming the GDN-2 result | <90% → engineering bug, not a research finding |
| P4 | **State-traffic reduction:** measured DRAM bytes/token on `S` drop **≥1.8×** vs the decomposed/naive path | <1.5× → the algebraic 5→2-pass restructuring is incomplete |
| P5 | **Correctness:** max-abs error vs `fla` fp32 reference **< 2e-2** in bf16, *within* the reference's own seed-to-seed band, and **no drift** over 512 carried steps | drift grows with step count → state-update numerics bug |
| P6 | **Batch crossover (headline result):** map decoupling overhead `C(B)` = GDN-2 ÷ GDN decode latency across batch `B ∈ {1,4,16,64}`; predict **free below B\*** (`C ≤ 1.10×`, state-bound) and a **measurable, characterized cost above B\*** as arithmetic intensity crosses ~1 FLOP/B. "Free below B\*, costs `C(B)` above" is more general/publishable than the batch-1 point | no B\* in {1..64} (overhead flat across batch) → the "free-below-B\*" framing is wrong; report the actual `C(B)` curve regardless |
| P7 | **End-to-end (projected — no checkpoint exists):** compute the Amdahl ceiling first — `max e2e gain = (GDN-layer share of per-token decode time) × kernel speedup`, with the non-GDN 25% costed as **2K sliding-window attention** (the GDN-2 reference hybrid), not full attention. Then *optionally* validate on a **small self-trained GDN-2 proxy** affordable on rental. Label measured-vs-projected explicitly | projected ceiling < a few % → kernel not worth integrating (state the number, don't bury it); or proxy diverges from projection → integration / state-management bug |

**[INFERENCE]** Note the honesty baked in: the headline is **not** a large speedup multiplier. If `fla` is already near roofline, the contribution is (a) the **GDN-2-specific decode kernel** where none is optimized yet (the layer is ~3 weeks old), (b) the **measured free-decoupling result**, and (c) the **roofline characterization**. That framing survives an adversarial reviewer; a "10× over torch.compile" claim does not.

---

## 4. Scope — what this is NOT (the most important section)

- **NOT** a prefill or training kernel — NVIDIA's chunkwise Triton kernels and FlashQLA (TileLang) own those. Decode-step recurrence only.
- **NOT** a full model rewrite — only the GDN-2 linear-attention layers' decode path; the interleaved non-linear layers (the GDN-2 reference hybrid uses **2K sliding-window attention**, not full global attention) keep their existing windowed-KV kernels.
- **NOT** a batch-1-only result — include small/continuous-batch sizes relevant to real serving, but batch-1 is the cleanest memory-bound regime for the controlled study.
- **NOT** dependent on a trained checkpoint — **NVlabs released training code only; no weights exist.** The correctness/roofline study runs on the **released GDN-2 architecture** (config: 16 heads, `d_k=d_v=128`, 1.3B) with **random-init or a small self-trained state** — sound because numerics (vs the `fla` reference, on identical weights) and bandwidth do not need *trained* weights. End-to-end *quality* is out of scope until weights are released or a small proxy is trained.
- **NOT** at 35B scale — the 35B / 8×H100 validation is **dropped from scope** (out of a rental-GPU budget; see the execution plan). The 1.3B-config kernel on bursty rental H100 is the entire substrate.

---

## 5. Baselines (the panel's correction)

| Tier | Baseline | Role |
|---|---|---|
| Floor | naive PyTorch decomposed recurrence | shows the 5+-pass HBM disaster; lower bound |
| Floor | `torch.compile` of the recurrence | weak floor only; **not** the headline comparison |
| **Real baseline** | **`fla` recurrent GDN/GDN-2 kernel** | the SOTA decode baseline to match/beat |
| Reference (correctness) | `fla` fp32 recurrent | ground-truth oracle, locked before kernel work |
| Out of scope | FlashQLA (TileLang, prefill) | prefill-optimized; not a decode comparison |
| Hardware ceiling | H100 achieved-DRAM-bandwidth roofline | the true ceiling; all claims are % of this |

---

## 6. Eval harness (locked before the kernel — Phase 1)

**6.1 Correctness oracle.** `fla` fp32 recurrent GDN-2, instantiated from the **released architecture** (trained weights *not* required — the oracle compares your kernel to the `fla` reference on **identical** weights, random-init included). Plus scalar-tied GDN as the degenerate check that the kernel recovers GDN when `b_t=w_t=β_t` and decay collapses — a free correctness test the architecture gives you.
- Single-step max-abs/rel error in bf16 and fp16.
- **Multi-step drift test:** carry the state for 512+ generated steps, plot error vs step; flat = pass, growing = state-update bug.
- **Adversarial numerics battery:** NaN/Inf inputs, empty/zero gates, extreme-magnitude keys/values, full-decay and zero-decay limits.

**6.2 Performance harness.**
- Lock clocks: `nvidia-smi --lock-gpu-clocks=tdp,tdp`.
- Nsight Compute per kernel: `dram__throughput.avg.pct_of_peak_sustained_elapsed`, achieved occupancy, registers/thread, SMEM/block, % of memory roofline.
- Nsight Systems for the decode timeline (launch overhead, gaps).
- Sweep: batch ∈ {1, 4, 16, 64}, the released 1.3B config (16 heads, `d_k=d_v=128`), dtype ∈ {bf16, fp16}; average over ≥1024 decode steps; report avg@3 across seeds.
- Measure DRAM bytes/token on `S` directly (P4).

**6.3 Vibe-eval (the bug-catcher unit tests miss) — same-weights divergence test.** No trained checkpoint needed: instantiate the released GDN-2 architecture with **fixed** weights (random-init or a small self-trained proxy), run a 100+-token decode with **your kernel vs the `fla`-reference path on identical weights**, and assert the token sequences match within tolerance. This tests kernel-vs-reference *equivalence*, not generation quality — so it catches the state-carry bugs that pass single-step tests (they surface as divergence after N tokens) **without** trained weights.

**Exit criterion:** harness produces sane numbers on the reference *architecture* (random-init weights suffice) and is version-controlled, before any kernel optimization.

---

## 7. Kernel design (the build — Phases 5–6)

**7.1 The decode recurrence (per token, per head), GDN-2 form.** The exact NVlabs reference update is
`S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t−1} + k_t (w_t ⊙ v_t)ᵀ`,
with state `S ∈ ℝ^{128×128}`, `D_t = Diag(α_t)` (channel-wise decay), erase gate `b_t ∈ [0,1]^{d_k}` (key axis), write gate `w_t ∈ [0,1]^{d_v}` (value axis). Decomposed for the kernel:
1. **Decay:** apply `D_t` to `S_{t−1}` (absorbed into the rank-one factors per the WY form, not a separate full-state multiply).
2. **Erase:** apply the rank-one key-side projector `(I − k_t (b_t ⊙ k_t)ᵀ)` to the decayed state — GDN-2's asymmetric, channel-selective erase factor.
3. **Write:** add the value-side gated outer product `k_t (w_t ⊙ v_t)ᵀ`.
4. **Read:** `o = q_tᵀ S_t` (with the model's L2-normed q/k, short-conv, SiLU paths handled at the layer boundary, not inside the hot kernel).

**7.2 The optimization (the actual contribution).**
- Keep `S` **register/SMEM-resident** for the whole 4-step update — never round-trip to HBM mid-update.
- **Algebraic restructuring to 1 read + 1 write pass** over `S` per token (fuse decay+erase into the read, fold the write into the single write-back), the GPU analog of the USC five-phase→two-pass datapath, adapted to GDN-2's *decoupled* asymmetric factors.
- **Exploit grouped-value structure** (GVA/GQA-style head sharing) to share query/key datapaths across head pairs where the config allows, cutting redundant loads. *(USC 2603.05931.)*
- Tune SMEM layout / tiling / occupancy via the maker/verifier agent harness (§9); the human owns the roofline interpretation of which layout wins and *why*.
- **Triton-first.** The contribution is *traffic reduction* (passes over `S`), not tensor-core scheduling — Triton should reach the ≥85% memory-roofline target (P1) at ~5× less effort than CUDA. Drop to CUDA **only** if Nsight shows Triton leaving measurable bandwidth on the table (e.g. the 1-read/1-write restructuring not realized in the generated PTX).

**7.3 Correctness-before-performance.** No performance number is reported until §6.1 passes, including the 512-step drift test. A fast kernel that drifts is a wrong kernel.

**Exit criterion:** single-batch overfit analog → the kernel reproduces `fla` fp32 output within tolerance at step 1 *and* step 512; init/numeric sanity matches theory.

---

## 8. Ablation plan (Phase 10 — one variable at a time)

Each ablation isolates one axis; avg@3; kill a direction after 3 failed variants.
- SMEM layout of `S` (row- vs column-major vs swizzled for bank-conflict-free access).
- Pass structure: naive 5-pass → 3-pass → 2-pass (the restructuring, quantified against P4).
- Gate handling: scalar-tied (GDN) vs decoupled (GDN-2) at identical layout → **isolates the free-decoupling thesis P2**. Since NVlabs's ablation shows the **erase gate `b_t` accounts for most of the quality gain**, add the load-bearing sub-ablation: tie `w_t`, vary `b_t` alone, to price the erase path's decode cost specifically.
- Head sharing on/off (GVA datapath sharing).
- dtype and accumulation precision (bf16 state vs fp32 accumulation).
- Batch sweep to locate B\* (P6).

---

## 9. Agent-orchestration component (the 2026 signal)

Build via the maker/verifier harness — and **document the human-judgment layer**, because that distinction is the hiring signal.
- **Verifier-first:** the §6.1 correctness oracle + Nsight roofline harness are built *before* any agent writes a kernel. No layout is delegated without an automatic correctness+roofline gate.
- **Maker agent:** proposes SMEM layouts, tiling, occupancy/register configs; emits candidate Triton/CUDA variants.
- **Verifier agent:** runs the correctness battery + drift test + Nsight roofline + the adversarial numerics on every candidate; rejects on any failure; logs to an audited trace.
- **Orchestration:** Tier-2 parallel git-worktrees sweep layouts; Tier-3 overnight drains the long tail of occupancy configs.
- **Human-only steps (never delegated):** the hypothesis, the correctness-oracle design, the roofline *interpretation* of which variant wins and why, and the forensic postmortem of every dead end. The writeup states explicitly **what was delegated, what was refused, and why** — the deskilling-resistant discipline.

---

## 10. Kill criteria (commit before starting)

- **Step-0 dependency gate (resolve before any kernel code is written):** (a) confirm reliable rental-H100 access + a locked-clock measurement budget; (b) confirm the checkpoint story — **no NVlabs weights exist**, so commit to architecture-only correctness/roofline + either a *projected* e2e (P7 Amdahl) or a small self-trained proxy. If neither e2e path is affordable, P7 is **explicitly descoped to a projection** up front — that is a scoping decision, not a failure.
- 3 weeks wall-clock on layout search with **no variant clearing P3 (90% of `fla` recurrent)** on the GDN case → engineering bug, fix or stop.
- **Free-decoupling thesis P2 fails** (GDN-2 ≫ GDN at decode even with resident state) → this is a **publishable negative result** about the real cost of erase/write decoupling at decode; write the postmortem, don't bury it. *(Documenting killed ideas is top-1% behavior — cf. DeepSeek-R1's appendix listing MCTS/PRM as explicitly killed.)*
- Correctness drift (P5) uncloseable → stop; a fast wrong kernel is worthless.

---

## 11. Reproducibility (Phase 12 — release discipline)

Reproducible from artifacts on a **single (rental) H100**:
- Pin `fla` commit, the **NVlabs GDN-2 code commit + architecture config** (and the small-proxy checkpoint hash *or* the random-init seed, if used — there is no official checkpoint to pin), CUDA/cuDNN, Triton, PyTorch versions; the clock-lock command; the exact Nsight invocation.
- Ship the harness, the kernel, the config YAML, and a one-command repro script. The benchmark report is only releasable if a reader can regenerate every plot from the repo.
- **License note:** NVlabs GDN-2 code is **NVIDIA Source-Code-NC** (non-commercial). This kernel and any derived weights inherit that constraint — fine for a public portfolio/research artifact, **not** for production shipping. State this in the repo README.

---

## 12. Deliverable artifacts

1. The decode kernel (Triton and/or CUDA) + integration into the GDN-2 layer's `generate()` path.
2. The maker/verifier agent harness (deterministic, audited) — itself a reusable portfolio piece.
3. This design doc (hypothesis → predictions → results-vs-predictions, filled in post-run).
4. The forensic postmortem (every dead end traced to mechanism: which SMEM layout bank-conflicted, which pass leaked to HBM, where drift came from).
5. The benchmark report: H100 roofline plots, % of measured-memory ceiling, DRAM-bytes/token, GDN-2-vs-GDN decode, the `C(B)` overhead curve with B\*, and the **Amdahl-projected** e2e (measured on a small proxy only if one is trained).
6. A bilingual (EN → VI) technical blog walking the whole thing from first principles, terms preserved.

---

## 13. Signal map — why a hiring committee reacts

- **Roofline/first-principles mastery** → P1–P7 framed as % of the memory-bound ceiling, not raw multipliers. *This is the dominant signal* — an inference team asks "do you understand why decode is memory-bound?" before "how fast is it?"
- **GPU performance (Req 4)** → the state-resident 2-pass kernel + the `fla`-relative result + correctness-before-speed.
- **Discipline + forensic debugging (Req 1/3)** → the design doc + the postmortem with mechanism-level failure traces.
- **Agent orchestration (Req 8)** → the maker/verifier harness + the explicit delegate/refuse account.
- **Inference-systems (Req 2)** → integration into a 3:1 GDN-2:SWA hybrid decode path with per-request state management + the Amdahl-bounded **projected** e2e (measured on a small proxy if trained).
- **Architecture depth (Req 9)** → correct GDN-2 WY/fast-weight recurrence, validated by the GDN-recovery degenerate test.
- **Writing (Req 7)** → the bilingual, reproducible, legible writeup.

**[INFERENCE]** A kernel/inference team at NVIDIA, an open-weights lab (Qwen/Moonshot-adjacent, Prime Intellect), or a frontier lab's inference org reads this and flips from "is this person good?" to "when can we talk?" — because it is a correct, novel, decode-regime contribution on the exact layer their production models are bottlenecked on, with the roofline understanding that separates a research engineer from a kernel typist.

---

### Appendix A — Primary sources
- **[VERIFIED]** GatedDeltaNet-2: Decoupling Erase and Write in Linear Attention — arXiv 2605.22791 (NVIDIA — Hatamizadeh, Choi, Kautz; 21 May 2026). Code: github.com/NVlabs/GatedDeltaNet-2 — **training-only, no released checkpoint**, NVIDIA Source-Code-NC license. Config: 16 heads, `d_k=d_v=128`, 1.3B / 100B FineWeb-Edu; hybrid uses 2K sliding-window attention.
- **[VERIFIED]** A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode **on FPGA** — arXiv 2603.05931 (USC): batch-1 decode <1 FLOP/B, 5-phase→2-pass restructuring, GVA sharing. (GPU is the open analog this capstone targets.)
- **[UNVERIFIED]** ONNX issue #7689 (naive GDN decode ~5 HBM passes/token, fused-gap figure, 75% GDN layers): cite **not confirmed here** — the HBM-pass *mechanics* are independently sound, but the per-head state for the released config is **32 KB** (512 KB/layer), not "512 KB/head".
- **[UNVERIFIED]** FlashQLA (qwen.ai/blog): TileLang prefill kernels, reported ~2–3× fwd over `fla` Triton on Hopper — **verify before citing**. Out of scope (prefill, not decode).
- **[VERIFIED]** flash-linear-attention (`fla`, fla-org GitHub): canonical Triton baseline; an explicit GDN-2 dependency.
- Kimi Delta Attention (KDA) — arXiv 2510.26692: channel-wise decay + scalar gate; the predecessor GDN-2 generalizes (cited in the NVlabs acknowledgements).
- Raschka, LLMs-from-scratch ch04/08 & Gated DeltaNet writeup: ~3:1 hybrid pattern; KDA channel-wise gating.


---

# Part II — Execution plan (4-week barbell)


> **Provenance:** written 2026-06-14 after fact-checking the premises against primary sources
> (arXiv 2605.22791 + the NVlabs repo). Governing constraints: **(1) rental H100 budget** (bursty hours,
> not a standing cluster; no 100B-token pretrain), and **(2) target role undecided** at planning time.

---

## 0. The decision, in one paragraph

Those two constraints *change the answer* from "pick the deepest spike" to **build a barbell**: one
high-variance deep artifact that makes a specific team say *"when can we talk"*, plus a low-variance base
that survives any generalist screen and keeps every door open. Going all-in on the single most
specialized artifact when you don't yet know your target is the classic talented-engineer mistake —
DELTA is a bullseye for ~5–10 identifiable teams (Songlin Yang / `fla`, NVIDIA's linear-attention group,
Qwen/Moonshot inference) and semi-illegible to everyone else. So you pair it. Rental budget is *exactly*
enough for DELTA's real scope, because its GPU need is **bursty** (lock clocks → Nsight → batch sweep →
done), not a standing cluster — and the base track (RL) is nearly free.

- **Spike = DELTA.** The corrected-scope decode kernel: correctness oracle + measured roofline +
  free-decoupling thesis + the `C(B)` batch-crossover. ~20–40 H100-hours total. The differentiator.
- **Base = A5 (GRPO/Dr.GRPO "aha") + the A2 finish.** A5 is the *other* scarce-2026 cluster (stable RL
  with a verifiable reward) and is CPU/near-free; the A2 finish (`fla` baseline + Nsight harness +
  tightened kernel + the 100B-memory one-pager) **is literally DELTA's Phase-1 infrastructure**. Spike and
  base share connective tissue rather than compete.
- **The role question resolves itself:** finish this and you have signal for inference/systems **and**
  post-training. Decide your target later, based on which track you were better at.

---

## 1. Compute discipline (the rental constraint forces the right habit)

The single rule: **scaffold and debug on CPU + a cheap 4090 spot; buy H100 hours only for locked-clock
measurement.** This is not a limitation — it's the discipline a frontier perf engineer uses anyway.

| Phase of work | Where it runs | Why |
|---|---|---|
| Harness, oracle, correctness battery, agent scaffolding | **CPU / free** | numerics + plumbing need no GPU |
| Kernel authoring, single-step debug, drift test (short) | **4090 spot / cheap** | functional iteration, not measurement |
| Locked-clock roofline (P1), batch sweep (P6), final avg@3 ablations | **rental H100, bursts** | the *only* numbers that must be on-target hardware |
| A5 RL ("aha" on Countdown/math, Qwen2.5-1.5B) | **CPU dev + ~1 cheap GPU burst (~$30–100)** | the canonical cheap RL proof |

**Budget target:** DELTA's entire headline (P1–P6) fits in ~20–40 H100-hours — a few hundred dollars.
Never run an H100 to *debug*; debug on the 4090, then spend H100 time only to *measure*. Lock clocks
(`nvidia-smi --lock-gpu-clocks=tdp,tdp`) and pin the SKU before any number counts.

---

## 2. Step 0 — dependency gate (before any kernel code; opened only once the A5 base has shipped)

This mirrors RFC §10's Step-0 gate. Resolve **all three** or DELTA's scope changes *before* you start, not
after you've sunk a week:

1. **H100 access confirmed** — a provider, a price, a locked-clock test run. (You have budget; pick the
   SKU now — note SXM vs PCIe / HBM3 vs HBM2e so P1's denominator is the right one.)
2. **Checkpoint story decided** — there is **no NVlabs checkpoint** (training code only). Commit to:
   architecture-only correctness/roofline (random-init, sound) **+** a *projected* e2e (P7 Amdahl). Decide
   now whether you'll also train a small proxy for a *measured* e2e, or leave P7 projection-only. Either is
   defensible; deciding up front is the discipline.
3. **`fla` GDN/GDN-2 recurrent baseline runs** — clone `fla` + the NVlabs GDN-2 code, get the reference
   recurrent decode producing output on CPU/4090. This is your oracle and your P3 baseline; nothing
   proceeds until it runs.

**Exit:** all three green, written into the RFC's §10 as committed. If H100 access or budget falls
through → flip to base-first (finish A2 systems + A5), do DELTA when access is solid.

---

## 3. The 4-week schedule

> Maps to the frontier 12-phase workflow and the RFC predictions. **The DELTA sprint is base-first: it
> starts only after the A5 RL "aha" ships and Step-0 clears** (see [`STRATEGY.md`](STRATEGY.md) §8). Weeks
> below are **relative** (Sprint W1–W4), not calendar dates — anchor them to your real start. Each week
> has **one** headline question — no multiplexing.

### Sprint Week 1 — Lock the harness (RFC Phase 1 = your A2 inference-systems finish)
**Headline question: "Does my correctness oracle + roofline harness produce sane numbers on `fla` before I write a single optimized line?"**
- Build the **correctness oracle** (RFC §6.1): `fla` fp32 recurrent vs a naive reference, on identical
  random-init weights; single-step max-abs/rel error (bf16/fp16) + the **512-step drift test** + the
  adversarial numerics battery (NaN/Inf, zero/empty gates, full/zero decay).
- Build the **Nsight roofline harness** (RFC §6.2): locked clocks, the STREAM/empty-copy microbench that
  *measures* achieved DRAM bandwidth on your SKU (this is P1's denominator), the per-kernel
  `dram__throughput…pct_of_peak_sustained_elapsed` capture, DRAM-bytes/token on `S` (P4).
- **A2 connective tissue:** this harness *is* the A2 "make one GPU fast" deliverable. In the same week,
  tighten your existing FA2 (currently 53% of SDPA) or write one clean memory-bound microkernel toward
  roofline — rebuild kernel confidence on known ground *before* the novel recurrence. Write the 100B-memory
  one-pager (the "train a 100B model" interview answer) — it's free and closes A2.
- **Exit criterion (RFC §6 exit):** harness green on the reference architecture, version-controlled, *before*
  any kernel optimization. This is the phase non-frontier people skip; you do not.

### Sprint Week 2 — Correct kernel, Triton-first (RFC Phases 5–6)
**Headline question: "Does my fused, state-resident kernel reproduce `fla` fp32 at step 1 AND step 512?"**
- Write the fused decode kernel in **Triton first** (RFC §7.2): keep `S ∈ ℝ^{128×128}` register/SMEM-resident
  across the full decay→erase→write→read update; implement the exact NVlabs recurrence
  `S_t = (I − k_t(b_t⊙k_t)ᵀ) D_t S_{t−1} + k_t(w_t⊙v_t)ᵀ`.
- **Correctness before performance** (RFC §7.3): no speed number until the 512-step drift test is flat and
  the degenerate GDN-recovery check (b=w=β, decay collapsed) passes. A fast kernel that drifts is a wrong
  kernel.
- Stand up the **maker/verifier harness** (RFC §9) *verifier-first* — but **time-box it** (≤2 days). It earns
  its keep by gating layouts behind the auto correctness+roofline check, not by becoming a second project.
- **Exit:** kernel reproduces `fla` fp32 within tolerance at step 1 and step 512; P5 met.

### Sprint Week 3 — Roofline + the free-decoupling result (RFC §8 · Phase 10)
**Headline question: "What % of the measured memory roofline, and is decoupling free below B*?"**
- Burst H100 for the real numbers: P1 (≥85% of *measured* roofline), P3 (≥90% of `fla` recurrent on the GDN
  case *before* claiming GDN-2), P4 (≥1.8× state-traffic reduction).
- Run the **headline ablation** (RFC §8): scalar-tied (GDN) vs decoupled (GDN-2) at identical layout → P2;
  then the load-bearing sub-ablation — tie `w_t`, vary `b_t` alone (NVlabs shows the **erase gate carries
  most of the gain**), to price the erase path specifically.
- **`C(B)` batch crossover (P6 — now a headline result):** sweep B∈{1,4,16,64}, map overhead
  C(B)=GDN-2/GDN latency, find B*. "Free below B*, costs C(B) above" is the publishable finding.
- avg@3 across seeds on every number that goes in the report.
- **Exit:** P1–P4, P6 measured and written into the RFC's results column.

### Sprint Week 4 — Projected e2e, postmortem, writeup (RFC Phases 11–12)
**Headline question: "Can a stranger regenerate every plot, and is every dead end traced to a mechanism?"**
- **P7 projected** (no checkpoint): compute the Amdahl ceiling — GDN-layer share of per-token decode time ×
  kernel speedup, with the non-GDN 25% costed as **2K SWA** (not full attention). If you trained a small
  proxy in Step 0, add the measured point and label measured-vs-projected.
- **Forensic postmortem** (RFC §12.4): every dead end to mechanism — which SMEM layout bank-conflicted,
  which pass leaked to HBM, where any drift came from. Documenting killed variants is top-1% behavior.
- **Reproducibility** (RFC §11): pin `fla` commit + NVlabs code commit + config + seeds (no checkpoint to
  pin), clock-lock, Nsight invocation; one-command repro; the **NC-license note** in the README.
- **Bilingual writeup** (EN→VI) from first principles.
- **Exit:** repo green, report regenerates from artifacts, RFC's predictions-vs-results table filled in.

### Interleaved across all 4 weeks · the A5 base (CPU/near-free — your insurance)
- Wire the **mandatory RL logging first** (entropy + the three KLs + reward/length stats) — *before* the
  first GRPO step; an RL run without it is uninterpretable.
- Build SFT masked-CE (correct `response_mask`) → **GRPO + Dr.GRPO** (group-relative advantage; the
  std-norm/length-norm de-bias toggle) → the verifiable-reward grader.
- Reproduce the **R1-Zero "aha" on Countdown** with Qwen2.5-1.5B (~$30–100, one cheap GPU burst; it emerges
  at 1.5B, fails at 0.5B). This is the cheapest legible proof you can make RL converge *and explain why it's
  stable*. Do this in the gaps when DELTA is blocked on an H100 burst — it spends ~zero rental hours.

---

## 4. What you are explicitly NOT doing (and why)

- **No 100B-token GDN-2 pretrain / no released-checkpoint dependency** — out of rental budget and
  unnecessary: correctness + roofline need only the architecture. (RFC §4.)
- **No 35B / 8×H100 validation** — dropped from scope. The 1.3B-config kernel is the whole substrate. (RFC §4.)
- **No A3 (scaling) or A4 (data) beyond a token slice** — least scarce, lowest ceiling for *this* portfolio.
  Run a tiny A4 dedup slice only to exercise pipeline order; skip the leaderboards.
- **No CUDA unless Nsight forces it** — Triton-first; CUDA only if it shows measurable bandwidth left on the
  table. (RFC §7.2.)
- **No "10× over torch.compile" framing** — the honest contribution is the GDN-2 decode kernel (none exists),
  the measured free-decoupling result, and the roofline characterization. (RFC §3.)

---

## 5. Kill criteria (committed before starting — mirrors RFC §10, plus barbell-level)

- **Step-0 gate unmet** (no H100 access, or no checkpoint decision) → don't start the kernel; flip to
  base-first.
- **3 weeks on layout search, no variant clears P3 (90% of `fla` recurrent) on GDN** → engineering bug,
  fix or stop.
- **Free-decoupling thesis P2 fails** (GDN-2 ≫ GDN at decode even with resident state) → **publishable
  negative result** on the real decode cost of erase/write decoupling; write the postmortem, don't bury it.
- **Correctness drift (P5) uncloseable** → stop; a fast wrong kernel is worthless.
- **Barbell-level:** if DELTA consumes Week 4 and the A5 base hasn't started, **stop DELTA polish and ship the
  A5 "aha"** — finishing with *zero* post-training signal is the one outcome the barbell exists to prevent.

---

## 6. Definition of done — the portfolio after 4 weeks

**Spike (DELTA):** a public repo with the Triton decode kernel, the deterministic maker/verifier harness, the
RFC with its predictions-vs-results table filled in, the forensic postmortem, the benchmark report (measured
roofline %, DRAM-bytes/token, GDN-2-vs-GDN, the `C(B)` curve with B*, projected e2e), and the bilingual
writeup. Reproducible on a single rental H100.

**Base (A5 + A2):** the A2 systems finish (FA2 tightened, the harness, the 100B-memory one-pager) and a clean
GRPO/Dr.GRPO repo with the R1-Zero "aha" reproduced and the mandatory RL logging — the proof you can make RL
converge and explain it.

**The combined signal:** *I own the whole stack, I went deep on the decode kernel where the production wave is
bottlenecked, and I can make RL stable* — strictly stronger than either pure spike or pure breadth, and it
keeps both the inference/systems and post-training doors open until you choose.

---
