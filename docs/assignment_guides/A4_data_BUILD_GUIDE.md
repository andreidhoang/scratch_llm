# A4 — Data · BUILD GUIDE (CS336 → `scratch_llm` L4 Data)

> **📖 Read first (slides → this build):** Lectures **13 → 14** — what-data / disclosure (py) ·
> HTML→text + curation pipeline (py); then **12** (evaluation, py) for quality/contamination. Full
> map + read-order: [`../LECTURE_MAP.md`](../LECTURE_MAP.md).

> **One-liner.** CS336 A4 ("Filtering Language Modeling Data") is a CommonCrawl → **extract → filter →
> quality-classify → dedup** pipeline that turns raw web HTML into a trainable corpus. In `scratch_llm`
> this is **Layer L4 (Data)**, built in **`src/scratch_llm/data/`**. The mastery payload is two-fold: the
> **dedup machinery** (exact line dedup + MinHash/LSH fuzzy dedup) and the **trained quality classifier** —
> where the lesson is *signal design*: "quality" has no ground truth, so it is **defined operationally by
> the label source** you choose. Everything else (HTML→text, language-ID, PII regex, NSFW/toxic classifiers,
> Gopher heuristics) shares one `(label, score) + threshold` shape — internalize it once, then move on.
>
> **Status:** `data/` is a clean stub (`data/__init__.py` only) — this whole layer is **TO BUILD**.

---

## 1. What CS336 actually requires (every deliverable)

Every adapter name and point value below is quoted from the A4 PDF (`Problem` headers); the `tests/adapters.py`
hook is in brackets. Re-tagged against the real assignment — three tiers only: **LOAD-BEARING** (own it cold),
**COURSE-ROTE** (build to green, don't extend), **SKIP** (capped run / infra spend, no mastery carry).

| Deliverable (PDF `Problem`) | PDF § | Adapter / test fn | Est. | Priority |
|---|---|---|---|---|
| `look_at_cc` (4 pts) — eyeball WARC/WET; annotate **25** WET records (language, domain, page type); "how many until a high-quality page?" | §2.1 | (written, no adapter) | 1 h | **COURSE-ROTE** — but *do the 25-doc annotation*: "become one with the data" is the real lesson (web data is mostly garbage). |
| `extract_text` (3 pts) — HTML bytes → text via `resiliparse.extract.html2text.extract_plain_text`; encoding-detect via `resiliparse.parse.encoding.detect_encoding()` (UTF-8 may fail) | §2.2 | `[run_extract_text_from_html_bytes]` · `test_extract_text_from_html_bytes` | 2 h | **COURSE-ROTE** — library call; iterate WARC records with `fastwarc.warc.ArchiveIterator`. |
| `language_identification` (6 pts) — fastText `lid.176.bin`; return `(lang, score)`; en/zh re-map | §2.3 | `[run_identify_language]` · `test_identify_language` | 2 h | **COURSE-ROTE** — the *first* instance of the `(label, score) + threshold` filter pattern; learn the shape here. |
| `mask_pii` (3 pts) — emails → `\|\|\|EMAIL_ADDRESS\|\|\|`, phones → `\|\|\|PHONE_NUMBER\|\|\|`, IPv4 → `\|\|\|IP_ADDRESS\|\|\|`; each returns `(new_str, n_masked)` | §2.4 | `[run_mask_emails]`/`test_mask_emails`, `[run_mask_phone_numbers]`/`test_mask_phones`, `[run_mask_ips]`/`test_mask_ips` | 2 h | **COURSE-ROTE** — regex; note it *transforms*, never *drops*. |
| `harmful_content` (6 pts) — NSFW + toxic/hate-speech via Dolma Jigsaw fastText models; `(label, score)` | §2.5 | `[run_classify_nsfw]`/`test_classify_nsfw`, `[run_classify_toxic_speech]`/`test_classify_toxic_speech` | 2 h | **COURSE-ROTE** — classifier-apply; identical shape to language-ID (tests are sanity checks only, not accuracy proofs). |
| `gopher_quality_filters` (3 pts) — Gopher (Rae et al. 2021) heuristic subset: word-count 50–100k, mean word length 3–10, <30% lines ending in `…`, ≥80% words with ≥1 alpha char | §2.6 | `[run_gopher_quality_filter]` · `test_gopher` | 2 h | **COURSE-ROTE** — cheap, transparent structural heuristics; returns a `bool`. Build, internalize, move on. |
| **`quality_classifier` (15 pts)** — train a **fastText** classifier on **trusted-positive** text (Wikipedia external-reference URLs) vs **random CommonCrawl** negatives; return `(label, score)` | §2.7 | `[run_classify_quality]` · `test_classify_quality` | 4 h | **LOAD-BEARING** — the **signal design** (positive = trusted-source-linked, negative = random-CC) *is* the lesson: quality is defined by the label source. The 15-point weight is the PDF's own emphasis. |
| **`exact_deduplication` (3 pts)** — count line frequency by **hash**, rewrite each file keeping only lines unique across the corpus | §3.1 | `[run_exact_line_deduplication]` · `test_exact_line_deduplication` | 2 h | **LOAD-BEARING** — the simplest dedup primitive; two-pass count-then-rewrite, hash keys for bounded memory. |
| **`minhash_deduplication` (8 pts)** — MinHash signatures + LSH banding → candidate pairs → **true-Jaccard confirm** → cluster-and-drop (keep one per cluster); NFD/lowercase/punct/whitespace normalization | §3.2 | `[run_minhash_deduplication]` · `test_minhash_deduplication` | 5 h | **LOAD-BEARING** — the fuzzy-dedup machinery; the `P[match]=Jaccard` + LSH S-curve core (see §2). |
| `filter_data` (6 pts) — **leaderboard**: filter 5000 CC WET files (parallel) to minimize **Paloma C4-100-domains** perplexity; **report per-filter discard %**; report runtime | §4 | (script; leaderboard PR) | 6 h+ | **SPLIT** — **LOAD-BEARING:** the pipeline *order* + per-filter discard accounting, exercised on a **token-budget slice**. **SKIP:** the full 5000-WET / 375 GB run (see §6). |
| `inspect_filtered_data` (4 pts) — 5 kept + 5 removed examples with justification; iterate the pipeline if warranted | §4 | (written, no adapter) | 1 h | **COURSE-ROTE** — the "look at what your filters did" discipline; do the annotation on your slice. |
| `tokenize_data` (2 pts) — GPT-2 tokenizer; append `<\|endoftext\|>`; serialize `np.uint16` | §4 | (script) | 0.5 h | **SKIP** at scale — only needed to feed the training leaderboard you are not running (see §6). |
| `train_model` (2 pts) — train GPT-2-small for **200K iterations** on the filtered data; report best C4-100 val loss | §4 | (leaderboard run) | GPU | **SKIP** — pure infra/GPU spend, no mastery carry (see §6). |

---

## 2. The equations / algorithms that matter

**(1) MinHash signature.** For a document's set of word n-grams `S = {s_1,…,s_n}` and `k` hash functions
`h_1,…,h_k`, `minhash(h_i, S) = min_j h_i(s_j)`. The signature is the vector
`[minhash(h_1,S),…,minhash(h_k,S)] ∈ ℝ^k`. **Key fact:** `P[minhash(h_i,S_1) = minhash(h_i,S_2)] =
Jaccard(S_1,S_2) = |S_1∩S_2| / |S_1∪S_2|`. So the fraction of agreeing signature rows is an unbiased
estimate of Jaccard, in **O(k) space** instead of O(|n-grams|). *Why it matters:* this is what makes
document-level dedup tractable at corpus scale — you never materialize the full n-gram sets, you compare
fixed-length signatures.

**(2) LSH banding (the S-curve).** Split the `k`-row signature into `b` bands of `r` rows (`k = b·r`).
Two docs are **candidate duplicates** if they collide in *any* band. The collision probability for a pair
with Jaccard `s` is `P_candidate = 1 − (1 − s^r)^b` — a tunable S-curve with its knee near `s ≈ (1/b)^(1/r)`.
More bands → higher recall, lower precision (more candidates to confirm). *Why it matters:* this converts the
O(N²) all-pairs comparison into a bucketed, near-linear pass; the **`(b, r)` split is the precision/recall
dial you must be able to justify**. Candidates are then confirmed with *true* Jaccard before being treated as
duplicates — LSH proposes, Jaccard disposes.

**(3) Cluster-and-drop.** After confirming pairs above the Jaccard threshold, take the **transitive closure**:
if A~B and B~C, then {A,B,C} is one duplicate cluster. Keep one document per cluster (randomly), drop the rest.
*Why it matters:* duplicate removal is a connected-components problem, not a pairwise one — dropping both members
of every confirmed pair would over-delete; clustering keeps exactly one representative.

**(4) Quality-classifier training signal.** Positives = text from **trusted-source-linked** pages (Wikipedia
external-reference URLs; the GPT-2/WebText "linked-by-Reddit-karma" idea is the same trick). Negatives = random
CommonCrawl. Train fastText; the output **score threshold** trades precision vs recall of "keep." *Why it matters:*
**"quality" is defined operationally by the label source — there is no ground truth.** Choosing the positive set
*is* the modeling decision; the classifier just amortizes that judgment over the whole corpus. This is the single
most transferable idea in A4.

**(5) The `(label, score) + threshold` pattern.** Every classifier filter — language-ID, NSFW, toxic, quality —
returns `(label, confidence)`; the filter keeps documents whose confidence clears a threshold. *Why it matters:*
one threshold-sweep harness covers all of them, and the cost/quality trade-off (a higher threshold discards more,
raising mean quality but shrinking the corpus) is the same knob everywhere. Recognize the shared interface and you
implement four filters as one.

**(6) Filtering pipeline order — cheap/destructive early, dedup last.** `extract → language-ID (drop non-target
first; cheapest, highest-volume cut) → Gopher heuristics (cheap structural) → harmful/NSFW (classifier) → PII mask
(transform, not drop) → quality classifier (most expensive) → **dedup last**`. Dedup runs on the *survivors*:
deduping before filtering wastes work on docs you'll discard anyway, and (for exact dedup) corrupts the line-frequency
counts that the dedup itself depends on. *Why it matters:* order changes both total cost and the final distribution;
**per-filter discard accounting** (the `filter_data` written deliverable) is the audit trail that lets you reason
about *which* filter shaped the corpus.

> **Optional eval-hygiene aside (NOT a core A4 deliverable — no adapter, no points).** A4 has **no
> contamination-detection problem**. The PDF's only train↔eval rule is in §4: you *may* use the Paloma C4-100
> validation set to *build* filters, but must **never copy validation data into the training set** ("the language
> model should never see any data from the validation set"). If you ever want to *verify* that separation, the same
> n-gram/Jaccard primitive from (1)–(2), run **across** the train/eval boundary instead of within-corpus, is a standard
> overlap check — useful general eval hygiene, but explicitly out of scope for A4.

---

## 3. Map to `src/scratch_llm/data/`

L4 lives in **`src/scratch_llm/data/`** (today: `data/__init__.py` only — a clean stub). The build mirrors the
official `cs336_data/` scaffold and is verified against `../../../lectures/assignment4-data/tests/adapters.py`.
Per [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) (§ "A4 · Data"), organize it as **dedup/n-gram
primitives + a `curate` surface** (the filters and the quality classifier) — keep the load-bearing dedup machinery
clean and well-tested; treat the classifier-apply filters as a uniform `(label, score)` family behind one interface.

| CS336 deliverable | → `src/scratch_llm/data/` | Build emphasis |
|---|---|---|
| `exact_deduplication` | dedup primitive (hash-count line dedup) | **LOAD-BEARING.** Two-pass count→rewrite; hash keys for bounded memory. Test: a known-duplicate corpus collapses to the exact unique-line count. |
| `minhash_deduplication` | dedup primitive (MinHash + LSH + Jaccard confirm + cluster) | **LOAD-BEARING — the core.** Verify `P[match]=Jaccard` on a hand-built pair *before* trusting the LSH bucketing; expose `(num_hashes, num_bands, ngrams, jaccard_threshold)`; justify the `(b, r)` split via the S-curve. |
| `quality_classifier` | `curate` — the trained quality classifier | **LOAD-BEARING (signal design).** The deliverable is the *training-signal choice* (trusted-positive vs random-CC), not CC-scale throughput. Train on a subsample; the threshold is the precision/recall lever. |
| `gopher_quality_filters` | `curate` — cheap structural filter | **COURSE-ROTE.** A transparent `bool` heuristic; the cheapest quality cut, runs early. |
| `language_identification` / `harmful_content` (NSFW, toxic) | `curate` — classifier-apply filters | **COURSE-ROTE.** Implement once as the shared `(label, score) + threshold` interface; a single threshold-sweep harness covers all three. |
| `extract_text` / `mask_pii` | `curate` — ingest + transform | **COURSE-ROTE.** `extract_text` is the corpus entry point (HTML bytes → text); `mask_pii` transforms in place (never drops). |
| `filter_data` (pipeline order + discard accounting) | `curate` — the pipeline driver | **LOAD-BEARING discipline (slice only).** Encode the §2(6) order; log per-filter % discarded as the audit trail. **SKIP** the full 5000-WET run. |

**Scaffold pointers.** The official scaffold to implement against:
`../../../lectures/assignment4-data/cs336_data/` (empty module — write from scratch) and
`../../../lectures/assignment4-data/tests/adapters.py` (the 11 hooks above). The training half
(`cs336-basics/`, the Paloma run) is the SKIP leaderboard, not part of the `data/` build.

---

## 4. The frontier-2026 lens

**Commoditized (table-stakes, don't over-invest).** HTML→text extraction, language-ID, PII regex, NSFW/toxic
classifier-apply, basic Gopher heuristics — all library calls or well-trodden patterns. Full CommonCrawl-scale
processing (Slurm, 375 GB) is an infrastructure exercise, not a research one. Mastery here is *recognizing the
shared `(label, score) + threshold` shape* and the pipeline-ordering reasoning, not re-deriving each filter.

**Scarce (where the leverage is).**
1. **Dedup at scale — MinHash/LSH.** The `P[match]=Jaccard` + S-curve machinery is the one piece of A4 that is a
   genuine algorithm with a precision/recall dial you must defend. It recurs everywhere large corpora are built
   (RefinedWeb, FineWeb, Dolma all lean on near-dup removal), and "explain MinHash+LSH and pick the band count"
   is a standard data-engineering interview probe.
2. **Quality-signal design.** "Data, not compute, is the bottleneck" is the quiet 2026 consensus. **DataComp-LM**
   (Li et al., 2024, arXiv 2406.11794) showed the *filtering recipe* beats *more tokens*; **data-constrained
   scaling** (Muennighoff et al., 2023, arXiv 2305.16264) showed token *quality/repetition* changes the scaling
   *exponent*, not just the offset. The A4 quality classifier is the hands-on version of that argument: the
   trusted-positive label source you pick **is** the recipe.

**Frontier JD language this maps to.** "Build data pipelines for LLM training" (OpenAI); "curate critical
post-training data — SFT, RLHF" (Meta MSL); and the interview prompts **"how does data quality change a scaling/
training outcome?"**, **"walk me through MinHash + LSH dedup and how you'd set the band count,"** and (the optional
aside) **"how do you prevent train↔eval contamination?"**

---

## 5. Prioritization verdict — what matters / what to skip

**The ~20% that is LOAD-BEARING (own it cold):**
1. **The dedup machinery** — `exact_deduplication` (hash-count line dedup) **+** `minhash_deduplication`
   (MinHash signatures → LSH banding → true-Jaccard confirm → cluster-and-drop, with NFD/lowercase/punct
   normalization). The `P[match]=Jaccard` identity, the LSH S-curve, and the `(b, r)` precision/recall dial.
2. **The quality classifier (15 pts)** — train fastText on **trusted-positive vs random-CC**. The deliverable
   is the **signal design** (quality is defined by the label source), not CC-scale throughput.
3. **Pipeline order + per-filter discard accounting** — cheap/destructive-early, dedup-last (§2(6)); run the
   `filter_data` pipeline on a **token-budget slice** and log what each filter discarded. This is the `filter_data`
   *discipline* without the *leaderboard run*.

**COURSE-ROTE (build to green, do not extend):** `extract_text`, `language_identification`, `mask_pii`,
`harmful_content` (NSFW + toxic), `gopher_quality_filters` — internalize the `(label, score) + threshold` pattern
shared by all classifier filters, then move on. `look_at_cc` / `inspect_filtered_data` — *do the annotation*; it is
the "become one with the data" lesson, even though it isn't reusable code.

**SKIP (capped runs / infra spend, no mastery carry):**
- The full **5000-WET / 375 GB `filter_data` leaderboard run** — run only a small slice to exercise pipeline order +
  discard accounting, then stop.
- `tokenize_data` + `train_model` — the **Paloma C4-100 200K-iteration training leaderboard**. Pure GPU spend; the
  data layer is the deliverable, not the trained model's perplexity.
- **Slurm / `submitit` orchestration** — `concurrent.futures` on a slice is enough; the cluster path is
  course-environment-specific.

**Interview leverage:** data-pipeline design (the ordering reasoning); MinHash/LSH dedup (the algorithm + the band
count); *"how does data quality change a scaling/training outcome?"* (DataComp-LM / data-constrained scaling); and
*"how do you prevent eval contamination?"* (the optional §2 aside).

---

## 6. Build checklist (ordered, with discipline gates)

1. **Pre-read** PDF §3.1–3.2 (dedup) and §2.7 (quality classifier); know the three equations cold —
   `P[match]=Jaccard`, the LSH S-curve `1−(1−s^r)^b`, and the cluster-and-drop transitive closure — before any code.
2. **`exact_deduplication`** → green `test_exact_line_deduplication`. Two-pass: hash-keyed line counts, then rewrite
   keeping corpus-unique lines. **Gate (loss-at-init analogue):** assert a hand-built corpus with known duplicates
   collapses to the *exact* expected unique-line count.
3. **`minhash_deduplication`** → green `test_minhash_deduplication`. **Gate:** verify the `P[match]=Jaccard` property
   on a hand-built pair (known Jaccard) before trusting the LSH bucketing. Then confirm candidates with true Jaccard
   and cluster. **Justify the `(b, r)` band split** against the S-curve (write the target recall before tuning).
4. **The classifier-apply filters as one family** — `language_identification`, `harmful_content` (NSFW + toxic),
   behind a shared `(label, score) + threshold` interface → green `test_identify_language`, `test_classify_nsfw`,
   `test_classify_toxic_speech`. One threshold-sweep harness for all.
5. **`extract_text`** (`test_extract_text_from_html_bytes`), **`mask_pii`** (`test_mask_emails`/`test_mask_phones`/
   `test_mask_ips`), **`gopher_quality_filters`** (`test_gopher`) → all green. Note which *drop* vs which *transform*.
6. **`quality_classifier`** → green `test_classify_quality`. **Predict before you run:** state your trusted-positive
   source and the expected effect of raising the score threshold (more discarded, higher mean quality, smaller corpus)
   *before* fitting. The signal-design choice is the deliverable; record it (→ §7 ADR).
7. **Pipeline order + discard accounting (slice).** Wire the §2(6) order over a **token-budget slice** of WET data;
   log per-filter % discarded. **Predict before you run:** write the rough discard fractions you expect per stage,
   then compare. This is the `filter_data` discipline; do **not** launch the full 5000-WET run.
8. **`look_at_cc` / `inspect_filtered_data`** — annotate the 25 raw WET docs and 5 kept + 5 removed from your slice,
   with a one-line justification each. Iterate the pipeline if the annotation surprises you.
9. **Green-CI commit** — `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`; conventional scope:
   `data: <imperative>` (e.g. `data: add MinHash/LSH document dedup`). Each module docstring states its intent, the
   invariant it must satisfy, and (where relevant) the interview question it answers.

---

## 7. ADR triggers (log in `../adr/`, don't relitigate inline)

- **Quality-classifier trusted-positive source.** CC uses Wikipedia-external-reference URLs as the positive set; if
  your slice differs (or you subsample differently), the choice of positive source — and the negative sampling — *is*
  the modeling decision. → **ADR before training the classifier.**
- **Dedup aggressiveness — Jaccard threshold + `(b, r)` band split.** The S-curve knee, the confirm threshold, and how
  hard you dedup all trade near-duplicate removal against corpus coverage; too aggressive erases legitimate variety.
  → **ADR** (record the chosen `(num_hashes, num_bands, ngrams, jaccard_threshold)` and the target recall).
- **n-gram size + normalization for MinHash.** n-gram length (in words) and the NFD/lowercase/punct/whitespace
  normalization set jointly determine what counts as "near-duplicate." Too short floods candidates; too long misses
  paraphrase. → **ADR / design note.**
- **Pipeline order + per-filter thresholds.** The §2(6) order is the default, but the exact thresholds (language-ID
  confidence, NSFW/toxic cutoffs, quality score) set both cost and the final distribution. → **design note** with the
  discard accounting from the slice run as evidence.
- **Scope guard.** Keep MoE-data and multimodal-data curation as ADR stubs only — out of scope for the `data/` layer
  (CLAUDE.md frozen scope). The full Paloma training leaderboard is a SKIP, not a deferred ADR.
