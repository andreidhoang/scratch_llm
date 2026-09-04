# A3 §4.2 — tcgen05 / UMMA GEMM (sm_100a) — compile evidence

Kernel: `tcgen05_gemm_sm100a.cu` (1-SM `cta_group::1`, f16 in / f32 TMEM accumulator).
**Runtime correctness DEFERRED to the B200 rental day** — this compiles for sm_100a on the
sm_120 dev box but cannot run here (no tcgen05 / TMEM / cta_group on consumer Blackwell).

## Gate 1 — the exact pre-registered command (PTX)
```
nvcc -arch=sm_100a -ptx performance/rental/kernels/tcgen05_gemm_sm100a.cu -o /tmp/tc.ptx
# exit 0
grep -c tcgen05 /tmp/tc.ptx   # 21
```
tcgen05 opcodes present in the emitted PTX:
`tcgen05.alloc`, `tcgen05.relinquish_alloc_permit`, `tcgen05.mma.cta_group::1.kind::f16`,
`tcgen05.commit…mbarrier::arrive::one`, `tcgen05.ld.32x32b.x4`, `tcgen05.wait::ld`,
`tcgen05.fence::{before,after}_thread_sync`, `tcgen05.dealloc`. Plus TMA
`cp.async.bulk.tensor.2d…mbarrier::complete_tx::bytes` and the mbarrier glue.

## Gate 2 (stronger, also run) — ptxas encodes it (cubin + SASS)
```
nvcc -arch=sm_100a -cubin performance/rental/kernels/tcgen05_gemm_sm100a.cu -o /tmp/tc.cubin  # exit 0, no diagnostics
cuobjdump -sass /tmp/tc.cubin
```
ptxas lowered the inline PTX to genuine Blackwell tensor-core SASS:
- `@UP3 UTCHMMA gdesc[..], gdesc[..], tmem[UR5], tmem[URZ], idesc[URZ], UP1` — the
  `tcgen05.mma.kind::f16` with the **accumulator in TMEM** (the `tmem[UR5]` destination) and
  the input-D predicate (`UP1` = enable-input-d).
- `@UP1 UTCBAR [UR6], URZ` — `tcgen05.commit` arriving on the mbarrier.
- `UTCATOMSWS.FIND_AND_SET` / `UTCATOMSWS.AND` — TMEM alloc / dealloc.

## Structural self-review vs reference (gau-nernst tcgen05 · Colfax Part 3–4 · PTX ISA tcgen05)
1. **alloc** — `NCOL=256` cols (pow2, ≥32) from **warp 0 only**, `relinquish_alloc_permit` after. ✓
2. **TMA-stage A/B** — 2D bulk-tensor copy into SMEM, `expect_tx` + `try_wait.parity` mbarrier. ✓
3. **mma** — **single thread** (`tid==0`) issues `cta_group::1.kind::f16 [d-tmem],a,b,idesc,p`;
   `enable-input-d` false on the first K-atom (D:=A·B), true after (D+=A·B). ✓ → `UTCHMMA`.
4. **commit** — arrives on `bar_mma`, completion by mbarrier not `wait_group`. ✓ → `UTCBAR`.
5. **drain** — **full 128-thread warpgroup**; warp `w` OR's its lane-quadrant `(32*w)<<16` into the
   TMEM address so all 128 lanes are read. The 32-lane-per-warp rule is stated in a comment: a
   single-warp drain reads 32/128 lanes ⇒ 3/4 of the tile stale. ✓

## Honest deferrals (validated on the B200, not here)
- `idesc` is a `0` placeholder — the exact bit-fields (dtype/transpose/scale-ID) are rebuilt from
  CUTLASS UMMA traits on the rental day; only the *encoder path* is exercised here.
- Matrix-descriptor `LBO/SBO/swizzle` use the repo artifact's worked 128B-swizzle constants
  (`wgmma_descriptor_manual.md` §2: LBO=16, SBO=1024, swizzle=1); actual tile values come from the
  CuTe SMEM layout on the rental day. The `>>4` encoder itself is byte-checked against the artifact.
- The `tcgen05.ld` fragment → (row,col) epilogue mapping is reference-annotated; exact layout
  validated on the B200 (element-exact vs `torch.matmul`, per-lane-quadrant per the pitfall test).
- This is the **1-SM, correctness-first** rung only (no multi-stage pipeline / warp-specialization /
  persistent tiles / 2-SM `cta_group::2`) — those are the later B200-day steps
  (`../B200_day_runbook.md` §1; targets in `../../PERF_ENGINEERING_SPEC.md` §4 · A3).
- **No** claim of numerical correctness or TF/s from this box: gate is *compiles for sm_100a +
  PTX/SASS contains tcgen05/UTCHMMA + structural match*. Pre-registered target ~1,209 TF/s (1-SM
  early, ~54% of B200 dense BF16) is measured on the rental day.
