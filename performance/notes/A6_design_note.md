# A6 Design Note — Map the chattiest collective onto the fattest link

> A6 distributed, the "buildable-now" half (gloo/CPU correctness). The MEASURED half — busbw at line
> rate, MFU at 16/32/64 GPUs, the NVLink→IB cliff — is rental-gated (8×H200 serving day,
> `serving_day_8xH200_runbook.md`; optional Phase-5 multi-node). All `[FACT]` here are correctness, not
> bandwidth. Reuses `utils/comms_calc.py` (comms algebra) + `memory_math.py` (ZeRO 16Ψ), already shipped.

## 1. The four primitives, and why each is verifiable without the node

The parallelism taxonomy is DP · TP · PP · EP · CP. DP (DDP) + ZeRO + FSDP already exist (CS336 front).
A6 adds the serving-relevant three (TP, PP, EP) + the scoreboard (MFU) — and each has a correctness
oracle that runs on ONE box (gloo multiprocess on CPU, or pure arithmetic):

- **Tensor parallel (`tp_mlp.py`):** column-parallel GEMM-1 → GeLU → row-parallel GEMM-2 → **one
  all-reduce**, with the `f`/`g` conjugate autograd ops (`f`: identity fwd / all-reduce bwd; `g`:
  all-reduce fwd / identity bwd). Oracle: gloo output AND all grads == a same-seed single-process MLP
  (rtol 1e-5), and a wrapped process group counts **exactly 1 all-reduce/forward** (2 per fwd+bwd — the
  MLP's share of Megatron's 2-per-layer). Mutation-tested: double-counting the fc2 bias or dropping the
  all-reduce both break the oracle. TP is intra-node only (≤8, the NVLink domain) — that constraint is
  the whole reason it maps onto the fattest link.
- **Pipeline 1F1B (`pipeline_schedule.py`):** the schedule LOGIC (no GPUs). GPipe (all-F-then-all-B) vs
  **1F1B** (one-forward-one-backward steady state). Oracle: makespan == an independent Kahn longest-path
  over the dependency DAG (device order + `F(i,s)←F(i,s-1)` + `B(i,s)` needs `F(i,s)` & `B(i,s+1)`), and
  the **bubble fraction == the analytic `(p-1)/m`** (p=4,m=8 → 0.375; p=8,m=16 → 0.4375). The load-bearing
  1F1B win is **peak activation memory `min(p,m)` vs GPipe's `m`** — pinned by the sim, independent of m.
- **Expert parallel (`ep_moe.py`):** dispatch → expert-GEMM → combine via all-to-all. Oracle: gloo output
  == a single-process dense-gather reference (allclose), with **zero token-divergence** (a 1e-6 gate-logit
  diff can flip an expert — the routing decision is validated, not just the output). Verified to move real
  cross-rank traffic (not degenerate-local); mutation (drop the inverse-permute combine) is caught.
- **MFU/HFU (`mfu.py`):** `MFU = achieved-FLOP/s ÷ peak` via `C ≈ 6ND` (2 fwd + 4 bwd FLOPs/param/token;
  HFU adds recompute). Oracle: reproduces the published **PaLM 540B 46.2% MFU / 57.8% HFU** from its
  config (45.70% / 57.17% computed — a real external-number reproduction, not a tautology), plus a
  six-killer decomposition (unoverlapped comm · bubble · memory-bound kernels · small per-GPU batch · MoE
  imbalance · stragglers) whose shares sum to 1.

## 2. What the node adds (rental-gated), and why it's only measurement

Everything above is *structure* — correct on one box. The node adds the *numbers* that only a real
interconnect produces: NCCL all-reduce busbw approaching the 900 GB/s NVLink line rate (`busbw = algbw ·
2(n-1)/n`), the ring/tree crossover, MFU held at 16→32→64 GPUs, DeepEP/DualPipe overlap timelines, and
**the ~18× NVLink→IB cliff** (900 → ~50 GB/s crossing nodes — the single fact that decides every mapping:
put the chattiest collective on the fattest link, TP/EP intra-node, PP/DP inter-node). Per ADR-0012, A6
R0 (topology/busbw) + R1 (TP micro) run *inside* the 8×H200 serving day; the training-leaning depth
(multi-node EP, DualPipe, elastic ckpt) is optional Phase 5 — its one inference-relevant fact, the cliff,
is legible from the single-node numbers + nccl-tests, so the rental is a job, not a lesson.

**The A6 lesson, provable on one box:** the parallelism *primitives* (TP's 2 all-reduces, 1F1B's bubble,
EP's all-to-all, MFU's 6ND) are exact and testable in gloo/arithmetic — the *engineering* is which
collective rides which link, and that judgment is made from the interconnect table + the cliff, which the
node measures once. This curriculum arrives with the primitives verified and the mapping reasoned; the
rental confirms the bandwidth.
