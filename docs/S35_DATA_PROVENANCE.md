# S3.5 Data Provenance — what can and CANNOT be used in the scaling-law fit

## CRITICAL FINDING (2026-08-05)
The old `artifacts/s3_scaling_sweep/results.json` (s1-s7, the first sweep)
was produced with a tokenizer that was LOST when the original pod died and
could NOT be reconstructed byte-identically. The rebuilt tokenizer has a
different bytes/token ratio:

- OLD tokenizer: bytes/token = 4.08  (loss/bpb = 2.83)
- NEW tokenizer: bytes/token = 1.93  (loss/bpb = 1.34, matches nanochat)

Evidence: the SAME grid point s7 (d12, r8) yields val_bpb=0.940 in the old
sweep vs 0.386 in the new s7-batch8 rerun. bpb is tokenizer-invariant ONLY
when computed on the same val bytes under the same tokenizer — here the
difficulty per byte differs by ~2.4x.

## CONSEQUENCE
**Old s1-s6 are scientifically unusable in any joint fit with new data.**
`scripts/joint_fit.py::load_clean()` excludes them by construction (never
reads that path). Forcing them in would corrupt the exponents alpha/beta.

## CLEAN dataset (usable for the law fit)
All points use the rebuilt ClimbMix tokenizer (md5 4fc61379...) at eta*=0.0021:

| point              | N (M)  | D (B) | D:N | bpb    | source              |
|--------------------|--------|-------|-----|--------|---------------------|
| d12r4 (P5 winner)  | 135.29 | 0.541 | 4   | 0.4033 | phase 1             |
| d8r4 (width probe) | 59.26  | 0.237 | 4   | (pending) | phase 2          |
| s7 confirm         | 135.29 | 1.082 | 8   | (pending) | phase 2          |
| d14r8              | 195.23 | 1.562 | 8   | (pending) | phase 2          |
| d14r20             | 195.23 | 3.905 | 20  | (pending) | phase 2 (gated)  |

## Methodology note
At 4-5 clean points, the 4-param Chinchilla fit (E,A,alpha,B,beta) is
weakly identified. The PRIMARY estimators are the local-slope diagnostics
(2-param, robust): beta_local at fixed N=135M, alpha_local across the ladder.
The global fit is a sanity-check envelope, not the headline number. Bootstrap
CIs are reported WIDE and HONESTLY.
