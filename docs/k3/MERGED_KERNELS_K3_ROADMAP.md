# MERGED ROADMAP — Kernel Engineering × K3 Rebuild (one spine, 2026)

> Overlay, not new scope. Canon docs it fuses: `JOB_SPRINT/KERNEL_MASTERY_2026.md` (S1–S30
> kernel ladder), `docs/k3/ROADMAP.md` (K0–K10 K3 track), `docs/k3/FACTS.md` (claim ledger),
> `JOB_SPRINT/MASTERY_LADDER.md` (mastery ledger), `FAST_TRACK_2026.md` (money lanes).
> Nothing here re-sequences those documents' content — it interleaves them so every kernel
> session lands exactly when the K3 build needs it, and every K3 component doubles as a
> kernel-engineering rep. Authored 2026-08-09, operator request.

---

## 0. First principles — why the merge is the correct move

**One equation governs both tracks:**

```
time = max( FLOPs / peak_flops ,  BYTES / peak_bandwidth ,  latency_floor )
```

The kernel ladder teaches the three moves against this equation — (a) move bytes less
(tiling, fusion), (b) overlap the wait (pipelining, async, warp spec), (c) shrink the bytes
(quant, KV compression). **K3 is the case study where every architectural decision IS one of
those moves.** Read the architecture as a kernel engineer and it decomposes perfectly:

| K3 design choice | Which move it is | The number to know cold |
|---|---|---|
| KDA (linear attn, constant state) | (a)+(c): decode reads weights, not an O(N) KV cache | state = 96×128×128×2 B/layer, context-free |
| MLA latent KV | (c): KV bytes/token → 1.8% of MHA | 1152 B/token |
| NoPE | (a): KDA's gated delta rule IS the positional encoding — delete the RoPE bytes/flops entirely | Kimi Linear §6.1 |
| LatentMoE 0.5× | (c): experts compute/store at half width | inter 192 vs 512 latent in mini |
| MXFP4 QAT | (c): 4.25 bit/param | 1.561 TB for 2.78T params |
| AttnRes | (a): fuse residual-stream reads into one attention pass | overhead = L·d params |
| DSpark/EAGLE-3 draft | (b): overlap decode latency with speculation | 370 tok/s reported (vLLM) |

**Consequence:** a kernel learned in the abstract is trivia; the same kernel learned at the
moment K3 demands it is *co-design* — and co-design is verbatim the staff-engineer
inflection in 2026 postings (Anthropic: "co-design attention mechanisms for next-generation
hardware"). The merge converts 30 kernel sessions + 11 K3 phases from two competing
backlogs into one critical path.

**Market first principles (FAST_TRACK §0, verified):** CUDA-from-scratch carries the
highest skill premium (+35–85%); the interview language is CUDA C++; every documented
no-PhD frontier hire won with **measured numbers in public**. So the merge's output unit is
never "session completed" — it is a ledger number defended aloud.

## 1. The reasoning method — how to analyze hard, per node

Applied identically to every kernel session and every K3 rung. This *is* the HERMES
protocol, sharpened for the merge:

1. **Derive the prediction on paper first.** Roofline arithmetic: FLOPs, bytes, arithmetic
   intensity, which regime (memory / compute / latency). Write the predicted GB/s or TF/s
   *before* touching the GPU. No prediction, no run.
2. **Ask the five questions cold:**
   - What bytes move, from where to where? Can they shrink, move less, or overlap?
   - What is the oracle (cuBLAS, SDPA, `fla.ops.kda`, the float64 ref) and what is the
     parity contract?
   - What breaks silently? (numerics, α floors, absorption identity, accumulation traps —
     the silent-bug surfaces are the hand-built ones, per HANDCRAFTED.md)
   - What changes at the next scale? (sm120 → sm90: the ridge moves 130 → 295 FLOP/B; the
     same kernel flips regime)
   - What would the interviewer probe? (tiling choice, occupancy arithmetic, numerics
     defense — then defend it aloud, recorded, once a week minimum)
3. **Measure. The gap between prediction and measurement is the entire lesson.** A matched
   prediction confirms the model; a missed one means the mental model is wrong — find which
   term of the equation lied.
4. **Three-path equivalence everywhere it exists.** chunkwise == recurrent == float64 ref
   (KDA); block == full AttnRes; quant error accounting from config, not from books.
   Equivalence contracts are how you know you *understand*, not just *ran*.
5. **Ledger or it didn't happen.** Every number → `bench/RESULTS.md §K3` with hardware +
   method. Books vs tech report disagreements → FACTS.md (tech report wins).

## 2. The merged critical path — M0 → M6 (12 weeks, Lane-B ~25 hr/wk)

Kernel sessions are **node-pulled by K3 phases** — the same rule the book maps already use.
K2 (KDA) remains the critical path; d20-GQA runs regardless (decision of record 2026-08-02).

### M0 · Now — close the open loops (Week 0)
- Book the d20 8×H100 rental (the control-family anchor; also unblocks reasoningLLM's GPU band).
- CPU gates green; K0/K1 already banked (config.py, param_count.py, 2.78T closure).
- Reasoning rep: re-derive the d20 memory envelope (18,246 MiB @ B=16) from the equation, cold.

### M1 · "Bytes are all that matter" — KDA × memory-bound kernels (Weeks 1–3)
- **Build:** `core/kda.py` hand-built per K2 proposal — per-channel Diag(α), scaled sigmoid
  g_min=−5 (predict: α floor = e⁻⁵), conv k=4 + Swish, L2Norm, full-rank gate, chunk 64 WY/UT.
- **Kernel reps (fused, in order):** S1 RMSNorm → *the L2Norm/gate norms KDA just made you
  write*; S2–S3 softmax → online softmax (*AttnRes K4 and FA2 M3 both need it*); S4 GEMV →
  *decode is a GEMV — the KDA decode step is memory-bound on weights, this is that exact
  kernel*; S5 Top-K → *the router's top-k, foreshadows LatentMoE in M4*.
- **Gates:** three-path equivalence with per-channel α; GPU-day parity vs `fla.ops.kda`
  (fwd+bwd); S1–S5 numbers at 96–101% HBM defended.
- **Interview yield:** "why is decode memory-bound, and what does KDA do to the bytes?"

### M2 · "Feed the tensor cores" — KDA kernels × CUDA C++ gap (Weeks 3–5)
- **Build:** S6 Triton tiled GEMM → port KDA chunkwise training form to Triton against the
  reference; S7–S9 CUDA GEMM ladder (SMEM → WMMA → mma.sync + XOR swizzle) — the standing
  CUDA C++ blocker dies here, pulled by need, not by guilt; S10 `csrc/fundamentals/`.
- **Capstone pull-forward:** K10.2 — the **KDA decode kernel** (DELTA re-aimed at
  per-channel decay). Target ≥85% of memory roofline; correctness contract inherits from
  M1's three-path equivalence. This is the flagship public kernel artifact.
- **Gates:** mma.sync GEMM ≥80% cuBLAS; KDA decode kernel measured + defended; Triton
  chunkwise matches reference bitwise-tolerance.
- **Interview yield:** the whole CUDA C++ loop — tiling, swizzle, occupancy, defended live.

### M3 · Attention — Gated MLA × the FlashAttention ladder (Weeks 5–7)
- **Build:** `core/gated_mla.py` (K3 phase K3): full-rank sigmoid gate, NoPE mode, absorption
  identity re-verified under both. **Kernel reps:** S11–S13 (torch oracle → FA2 fwd → bwd),
  S14 paged decode with the MLA latent KV layout (1152 B/token — derive, then measure),
  S15 CUDA graphs (955 launches/token → why).
- **Reasoning reps:** why NoPE works (gated delta rule as data-dependent positional
  encoding — explain it from the recurrence, not the paper); why attention is memory-bound
  at decode and what MLA does to the ridge point; FA4 honesty flags (S19 dated claims —
  do not invent non-public numbers).
- **Gates:** absorption identity exact in float64 under NoPE+gate; KV-bytes accounting test;
  FA2 fwd ≥50% SDPA @4k (existing bar, re-defended on the new layout).
- **Interview yield:** "KV cache, FP8, speculative decoding" — the standard 2026 cold questions.

### M4 · MoE + numerics — LatentMoE × quant ladder (Weeks 7–9)
- **Build:** `situ.py` then `core/latent_moe.py` (K5): 0.5× latent, sigmoid router,
  Quantile Balancing vs F6 sign-step — both arms, pre-registered. `core/attn_res.py` (K4).
  Then **K6: assemble mini-K3** and train iso-FLOP vs d-series baseline (ClimbMix,
  S3 scaling gate already tripped → ratio-20 HOLD respected).
- **Kernel reps:** S29 MoE grouped GEMM + dispatch/combine (DeepEP/DeepGEMM study);
  S21–S25 numerics ladder fused with **K7 MXFP4 QAT** (`qat.py`: E2M1/block-32/E8M0 + STE,
  experts-only — exactly K3's scope). S24's FP8 accumulation trap (NIAH 91%→13%) is the
  war story that proves numerics literacy.
- **Gates:** balancing converges to q = mk/n without aux loss; |x| ≤ 100 post-SiTU;
  mini-K3 overfit-one-batch < 1e-2, val_bpb parity pre-registered; quant arithmetic
  reproduces 4.25 bit/param ⇒ 1.561 TB *derived from config*.
- **Interview yield:** MoE routing + load balancing + QAT — the exact Mistral/TM kernel reqs.

### M5 · Systems — serving the real checkpoint (Weeks 9–11)
- **Build:** K8 (1M-context memory math from config — KDA state constant + MLA latent KV vs
  full-attn baseline; honest eval: a 0.3B model teaches structure, not 1M capability).
  K9: 8×B300 Modal session, runbook already written — measure TTFT / tok/s / cold boot /
  $/M ourselves, three-way ledger vs book vs vLLM claims; test the fp8-KV question directly.
- **Kernel reps:** S26 continuous batching, S27 PD-disagg (+ vLLM v0.26 AFD line), S28
  speculative decoding (K10.1 DSpark-style draft on `mtp.py` if bandwidth allows).
- **Gates:** the honesty exercise landed in `bench/RESULTS.md §K3`; hybrid-state cache
  (KDA state + latent KV) serving mini-K3.
- **Interview yield:** "serve a 2.8T MoE" system design, answered with your own numbers.

### M6 · Convergence + the offer trigger (Weeks 11–12)
- Three public measured numbers exist now: **d20 loss curves + CORE**, **KDA decode kernel
  roofline %**, **K9 three-way serving ledger**. That is the Keller-Jordan-shaped artifact
  set, K3-flavored.
- Lane C triggers: applications out (Anthropic RL/inference, NVIDIA kernel, inference
  providers) per FAST_TRACK §2 — not before the numbers are public.
- K10 remainder (GGUF/A100-tier quant-of-quant, EP/EPLB notes) is opt-in, EV-ranked.

## 3. The weekly rhythm (non-negotiable skeleton)

| Block | Content |
|---|---|
| AM 30–45 min daily | Blank-page rebuild quarry — weakest-first ★ row from MASTERY_LADDER (K-rows now include k3/core as they land) |
| Deep block (Lane B) | Current merge node only: build → measure → ledger |
| 1×/week | Recorded mock from `challenges/nvidia/` — k_live → k_arch → k_serving; defends the week's node aloud |
| Ledger | `bench/RESULTS.md` per measured kernel; FACTS.md per claim; session log in KERNEL_MASTERY §4 |

Budget check (K3 ROADMAP §5): total cash to finish K0–K9 ≈ $200–350; the d20 rental is
separate Lane-B spend. Scale-to-zero discipline on every rental — $1,363/day is the
failure mode, not the plan.

## 4. What "mastered" means at the end of M6

- Every ★ K-row on the merged path at DEFENDED + NUMBERED (target: 30/89 total ladder by day 90).
- You can derive, cold: the decode memory wall, KDA state arithmetic, MLA absorption,
  4.25-bit accounting, MFU envelope — each with its measured number from *your* hardware.
- Three public artifacts with numbers, defended in ≥10 recorded mocks.
- The sentence you can say in the loop that almost no candidate can: *"I rebuilt the K3
  stack from scratch — KDA, gated MLA, LatentMoE, MXFP4 QAT — wrote the decode kernel to
  85% of the roofline, and served the real 2.8T checkpoint with my own measured TTFT and
  $/M. Here is the ledger."*

---

*Merge rule going forward: when a new kernel topic appears, it enters the ladder only when
a K3 (or d20 / DELTA) node pulls it. When a new K3 component appears, its kernel rep is
scheduled in the same week. One spine, one ledger, predict-before-you-run.*
