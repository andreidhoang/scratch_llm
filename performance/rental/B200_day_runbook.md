# B200 Day Runbook (Tier 3) — tcgen05 / TMEM / NVFP4-native MMA

> **Gate (ADR-0012):** enter only after the H100 day (Tier 1) is done and A3 §4.1 warp-specialized
> WGMMA is hitting ≥85% of H100 dense — the jump to B200 should only happen once Hopper headroom is
> exhausted. Target: 1× B200 (sm_100a, ~192 GB, ~8 TB/s), ~4–6 h, ~$25–45. **`-arch=sm_100a`** (the
> `a` is mandatory). RTX 5090/sm_120 is NOT a substitute — it has FP4 inference but **no tcgen05, no
> TMEM, no NVFP4-MMA, no `cta_group`** (00_foundations.md hard gate).

## 0. Pre-rental (do on the sm120/H100 box — don't burn B200 time on reading)

- [ ] Read Colfax CUTLASS tutorials Part 1–4 (WGMMA + UMMA + TMEM + NVFP4) — all four.
- [ ] Read the NVIDIA PTX ISA `tcgen05` section: `alloc`/`ld`/`st`/`cp`/`commit`/`fence`.
- [ ] Read the gau-nernst tcgen05 reference kernel (gau-nernst.github.io/tcgen05/, ~98% of B200 cuBLAS).
- [ ] A3 §4.2 tcgen05 PTX written + `nvcc -arch=sm_100a -ptx` compile-checked on the dev box (compiles,
      won't run).
- [ ] The A5 NVFP4 quantizer (`src/scratch_llm/quant/nvfp4_mxfp4.py`) ✅ verified in software on sm120
      (MSE 1.48× < MXFP4) — the B200 day only adds the *native MMA throughput*, the numerics are done.

## 0.1 Boot + verify

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv        # expect (10,0) sm_100
sudo nvidia-smi -lgc <max>,<max>                            # lock clocks
ncu --query-metrics | grep -i tmem                          # verify counters + TMEM metrics present
nvcc -arch=sm_100a -ptx trivial_tcgen05.cu -o /dev/null     # trivial tcgen05 compiles
```

## 1. A3 §4.2 — tcgen05 / UMMA GEMM, accumulator in TMEM

Pre-registered (PERF_PLAN Phase 3; B200 dense BF16 = 2,250 TF/s — quote DENSE, keynote figures are 2×
sparse):

| step | build | target `[INFERENCE, pre-reg]` |
|---|---|---|
| 1-SM | `tcgen05.alloc` (≥32 cols, one warp) → TMA-stage A/B → `tcgen05.mma.cta_group::1.kind::f16 [d-tmem]…` → `tcgen05.commit` on mbarrier → **full-warpgroup drain** (32-lane rule; single-warp drain ⇒ 3/4 of tile stale) | **~1,209 TF/s (~54% dense)** early |
| warp-spec | producer/consumer + persistent | **~1,300–1,476 TF/s (~58–65%)**; gau-nernst ref ≈98% cuBLAS |
| 2-SM | `cta_group::2` (MMA_M=256, leader-only issue) | honest **~8% win** (1209→1302) — SMEM-bandwidth relief, not raw math; >25% ⇒ double-issue bug |

Oracle: element-exact vs `torch.matmul` at fp32-accum tolerance; full 128-lane TMEM drain verified
(3/4-of-tile-stale is the classic bug). CUTLASS `cute::gemm` issues from a FULL warp — single-thread
`tcgen05.mma` deadlocks.

## 2. A3 §4.3 — NVFP4 block-scaled MMA

`nvf4` = block-16 + UE4M3 mantissa-bearing scale (vs MX block-32 + UE8M0); scales staged to TMEM via
`tcgen05.cp`, duplicated across 4 lane-quadrants; scale-IDs in the 32-bit idesc; `f8f6f4`-kind MMA.
Target: placement on the **9,000 TF/s dense FP4 tier**; accuracy vs BF16 on real weights (the software
NVFP4 numerics from A5 R3 are the correctness reference — the hardware only changes throughput, not the
values, so diff the native-MMA output against the A5 software `nvfp4_quantize` dequant).

## 3. A5 §7 — NVFP4 native element-rate vs FP8

Measure the hardware NVFP4-vs-FP8 element-throughput ratio (expect **~2× dense PFLOPS** FP4-vs-FP8 on
Blackwell — the hardware ratio; the ~2.3× e2e is a separate workload-dependent number, never conflate).
Records the one B200-only fact the sm120/H100 work cannot: the native FP4 tensor-core rate.

## 4. Before release

- [ ] tcgen05 GEMM %-of-dense + the 2-SM ~8% result logged; TMEM drain verified.
- [ ] NVFP4-MMA placement on the 9,000 TF tier + accuracy-vs-BF16 (diffed against the A5 software NVFP4).
- [ ] NVFP4:FP8 element-rate ratio logged (hardware, dense).
- [ ] DELTA note: NVFP4-on-recurrent-state probe (the capstone's sm_100a-gated headline) — if scoped, run
      the toy here; else record the gap.
- [ ] `git push`.

**Cut-order note (ADR-0012):** B200 is the first to drop under budget pressure — the DELTA NVFP4 headline
eventually needs it, but A1–A5 + Hopper kernels deliver ~80% of the program without it.
