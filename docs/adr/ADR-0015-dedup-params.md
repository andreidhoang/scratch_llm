# ADR-0015 — MinHash/LSH dedup parameters: `(num_hashes, num_bands, ngrams, jaccard_threshold)`

**Status:** accepted · 2026-07-03 (sprint node W7a, ADR-0014)
**Scope:** `src/scratch_llm/data/dedup.py` — the A4 fuzzy-dedup machinery. The A4 guide §7 names
this ADR trigger: the S-curve knee, the confirm threshold, and dedup aggressiveness trade
near-duplicate removal against corpus coverage.

## Context

`minhash_dedup` exposes the four dials the official adapter takes:
`(num_hashes k, num_bands b, ngrams n, jaccard_threshold t)`, with `k = b·r` (r rows/band).
Mechanics: a pair at true Jaccard `s` matches any one signature row with probability `s`
(the MinHash identity), so it becomes an LSH *candidate* with probability
`P = 1 − (1 − s^r)^b` — an S-curve with knee near `(1/b)^(1/r)`. Candidates are then confirmed
against **true** Jaccard ≥ `t` and clustered by union-find, so `(b, r)` only sets *recall* (which
true duplicates ever get checked) and candidate volume; precision is guaranteed by the confirm
step. False negatives are silent (duplicates survive); false candidates only cost confirm work.

## Decision

Pipeline default (exported as `DEFAULT_MINHASH_PARAMS`):

```
num_hashes = 100 · num_bands = 10 (r = 10) · ngrams = 5 (words) · jaccard_threshold = 0.8 · seed = 0
```

**S-curve justification (target recall pre-registered, then computed).** Target: catch
essentially all near-exact duplicates (`s ≥ 0.9`) while proposing almost nothing from unrelated
text (`s ≤ 0.5`). With `b = 10, r = 10` the knee `(1/10)^(1/10) ≈ 0.794` sits at the confirm
threshold 0.8, and `P_candidate(s) = 1 − (1 − s^10)^10` gives:

| s | 0.5 | 0.7 | 0.8 | 0.85 | 0.9 | 0.95 |
|---|-----|-----|-----|------|-----|------|
| P | 0.010 | 0.249 | 0.679 | 0.888 | **0.986** | **0.9999** |

i.e. ≥ 98.6% recall at `s ≥ 0.9`, ~1% noise at `s = 0.5`. Recall at exactly `s = t = 0.8` is
0.679 by construction — the knee *is* the threshold; borderline pairs are the ones we are most
ambivalent about dropping.

**Aggressive setting** (when higher recall at the threshold matters, e.g. license/boilerplate
removal — the official fuzzy fixture uses it): `num_hashes = 500, num_bands = 50` (same r = 10)
moves the knee to `(1/50)^(1/10) ≈ 0.676` and lifts recall at `s = 0.8` to 0.9966 (0.761 at
s = 0.7) at 5× signature cost. Keep `r = 10` and scale `b` — raising bands at fixed r only ever
*adds* recall (monotone in b; tested in `test_s_curve_recall_monotone_in_bands_at_fixed_k`).

**n-gram size + normalization.** `n = 5` words over the normalized form (NFD → lowercase → strip
combining marks + punctuation → collapse whitespace). Shorter n floods candidates with generic
phrase overlap; longer n misses light paraphrase. Verified on the official fuzzy fixtures: the
rails/react MIT licenses (differing in whitespace + attribution) land at true Jaccard ≈ 0.92
(≥ 0.8 → merged), while pytorch's 3-clause license sits at ≈ 0.007 — the 0.8 threshold separates
them by two orders of magnitude. Punctuation is *dropped* (not blanked to space); both variants
were measured and give indistinguishable fixture Jaccards (0.9186 vs 0.9191), so the choice
follows the guide's literal "strip punctuation".

**Determinism.** Hash functions are `(a·x + b) mod (2^61 − 1)` over 64-bit BLAKE2b base hashes
with coefficients from `random.Random(seed=0)` — never Python's salted `hash()`. Survivor per
cluster = lowest input index (first in input order), so a rerun on the same input list is
byte-identical.

## Consequences

- W7b's `filter_data` pipeline uses `DEFAULT_MINHASH_PARAMS` unless a caller overrides; dedup
  runs **last** (guide §2(6)) so signatures are computed only for survivors.
- Too-aggressive dedup erases legitimate variety: any tightening (lower `t`, more bands) must
  re-quote this table and state the new target recall first (predict-before-run).
- The `P[row match] = Jaccard` identity is enforced in-tree at 5σ binomial tolerance
  (`tests/test_data_dedup.py::test_row_match_probability_estimates_jaccard`); if a hash-family
  change breaks unbiasedness, that test — not the fixture tests — is designed to catch it.
