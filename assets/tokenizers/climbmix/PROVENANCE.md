# ClimbMix BPE tokenizer — canonical pinned copy

## Identity (must match across pod rebuilds)
- file: `tokenizer.json`
- md5: `4fc61379fc4bbaee73842a4aa8752a02`
- size: 1,206,438 bytes
- vocab: 32768
- bytes/token (measured on s7 val slice): 1.93

## Why this file is checked into git (not regenerated)
The `artifacts/f12_corpus_ablation/climbmix/tokenizer.json` and
`artifacts/s3_scaling_sweep/data/tokenizer.json` copies are gitignored and die
with their pod. Rebuilding via `scripts/rebuild_arm_tokenizer.py` is deterministic
in theory but NOT byte-verifiable after loss — a tiny library-version or
corpus-snapshot drift silently shifts every bpb measurement by ~2.4x (this is
exactly what corrupted the old s1-s7 sweep; see docs/S35_DATA_PROVENANCE.md).

This canonical copy under `assets/` is the single source of truth. New pods MUST
copy it into place and verify md5 before any run, rather than rebuilding.

**ENFORCED since 2026-08-31 — this used to be advisory prose only.** Until then
`scripts/run_s35_phase1.sh` *rebuilt* the tokenizer whenever it was absent, and skipped
verification whenever a file was present under any name — reproducing exactly the failure this
document describes. The pin is now executable: `scripts/_tokenizer_guard.sh` holds the md5 and
`require_pinned_tokenizer <dest>` copies-and-verifies, **hard-failing** if a file already sits at
the destination with different bytes. Both pipeline entry points (`run_s35_phase1.sh`,
`run_s3_sweep_climbmix.sh`) call it. A run that cannot prove it is using the pinned bytes will not
start.

## Where the code expects it
- corpus staging: `artifacts/s3_scaling_sweep/data/tokenizer.json`
- ablation arms:   `artifacts/f12_corpus_ablation/climbmix/tokenizer.json`

## Verify on a new pod (one-liner)
    md5sum assets/tokenizers/climbmix/tokenizer.json
    # must print: 4fc61379fc4bbaee73842a4aa8752a02
