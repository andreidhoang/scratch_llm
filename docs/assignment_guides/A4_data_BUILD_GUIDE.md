# A4 — Data · BUILD GUIDE (CS336 → reasoningLLM L4 Data)

> **One-liner.** CS336 A4 ("Filtering Language Modeling Data") is a CommonCrawl → extract → filter → dedup → train pipeline; in reasoningLLM this is **Layer L4 (Data)**, which feeds **`data/curation.py`** (reward/verifier-data curation) and **`data/contamination.py`** (train↔eval contamination). **Through-line tie:** the same dedup/quality machinery that makes a pretraining corpus clean is repointed at the *reward/verifier* dataset — the least-curated, highest-leverage data in the building — to ask *does cleaner reward data lower `hack_rate`?*; and the contamination check makes every VERA `true_quality_gap` number trustworthy (load-bearing, not dressing).

---

## 1. What CS336 actually requires (every deliverable)

Every adapter name and point value below is quoted verbatim from the A4 PDF (`tests/adapters.py` hooks in brackets).

| Deliverable (PDF `Problem` name) | PDF § | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|
| `look_at_cc` (4 pts) — eyeball WARC/WET; annotate 25 WET records; "how many until a high-quality page" | §2.1 | (written, no adapter) | 1 h | COURSE-ROTE (do the 25-doc annotation — it is the *real* lesson: web data is mostly garbage) |
| `extract_text` (3 pts) — HTML bytes → text via `resiliparse.extract.html2text.extract_plain_text`, encoding-detect via `detect_encoding()` | §2.2 | `[run_extract_text_from_html_bytes]` · `test_extract_text_from_html_bytes` | 2 h | COURSE-ROTE (use the library; not novel) |
| `language_identification` (6 pts) — fastText `lid.176.bin`; return `(lang, score)`; en/zh remap | §2.3 | `[run_identify_language]` · `test_identify_language` | 2 h | LOAD-BEARING-LITE (the (label, score) + threshold pattern is reused by every classifier filter) |
| `mask_pii` (3 pts) — mask emails → `|||EMAIL_ADDRESS|||`, phones → `|||PHONE_NUMBER|||`, IPv4 → `|||IP_ADDRESS|||`; each returns `(new_str, n_masked)` | §2.4 | `[run_mask_emails]`/`test_mask_emails`, `[run_mask_phone_numbers]`/`test_mask_phones`, `[run_mask_ips]`/`test_mask_ips` | 2 h | COURSE-ROTE (regex; not on the reasoningLLM critical path) |
| `harmful_content` (6 pts) — NSFW + toxic via Dolma Jigsaw fastText models; `(label, score)` | §2.5 | `[run_classify_nsfw]`/`test_classify_nsfw`, `[run_classify_toxic_speech]`/`test_classify_toxic_speech` | 2 h | COURSE-ROTE (classifier-apply; same shape as lang-ID) |
| `gopher_quality_filters` (3 pts) — implement the Gopher (Rae et al. 2021) heuristic subset | §2.6 | `[run_gopher_quality_filter]` · `test_gopher` | 2 h | LOAD-BEARING-LITE (cheap, transparent quality heuristics → reusable on reward-data text) |
| `quality_classifier` (15 pts) — train fastText quality classifier on Wiki-ref positives vs CC negatives; `(label, score)` | §2.7 | `[run_classify_quality]` · `test_classify_quality` | 4 h | COURSE-ROTE at CC scale; the *signal design* (positive=trusted-source, negative=random-CC) is LOAD-BEARING as a concept |
| `exact_deduplication` (3 pts) — count line frequency by **hash**, rewrite keeping unique lines | §3.1 | `[run_exact_line_deduplication]` · `test_exact_line_deduplication` | 2 h | **LOAD-BEARING** (the dedup machinery → `curation.py`) |
| `minhash_deduplication` (8 pts) — MinHash signatures + LSH banding + true-Jaccard confirm + cluster-and-drop; NFD/lowercase/punct normalization | §3.2 | `[run_minhash_deduplication]` · `test_minhash_deduplication` | 5 h | **LOAD-BEARING** (fuzzy dedup → `curation.py`; same n-gram/Jaccard core as contamination) |
| `filter_data` (6 pts) — leaderboard: filter 5000 CC WET files (parallel, `concurrent.futures`/`submitit`) to minimize **Paloma C4-100-domains** perplexity; report per-filter discard %; runtime | §4 | (script; leaderboard PR) | 6 h+ | SKIP at full scale (see §6); KEEP the pipeline-order + discard-accounting discipline |
| `inspect_filtered_data` (4 pts) — 5 kept + 5 removed examples w/ justification; iterate | §4 | (written) | 1 h | LOAD-BEARING-LITE ("become one with the data" — the Phase-3 discipline that A4.2 formalizes) |

---

## 2. The equations/algorithms that matter

**(1) MinHash signature.** For a document's set of word n-grams `S = {s_1,…,s_n}` and `k` hash functions `h_1,…,h_k`, `minhash(h_i, S) = min_j h_i(s_j)`. The signature is the vector `[minhash(h_1,S),…,minhash(h_k,S)] ∈ ℝ^k`. Key fact: `P[minhash(h_i,S_1) = minhash(h_i,S_2)] = Jaccard(S_1,S_2) = |S_1∩S_2| / |S_1∪S_2|`, so the fraction of agreeing signature rows estimates Jaccard in O(k) space instead of O(|n-grams|). *Why it matters:* makes dedup tractable at corpus scale; the **exact same n-gram + Jaccard core** is what `contamination.py` uses to detect train↔eval overlap.

**(2) LSH banding.** Split the `k`-row signature into `b` bands of `r` rows (`k = b·r`). Two docs are *candidate duplicates* if any band hashes identically. Collision probability for Jaccard `s` is `1 − (1 − s^r)^b` — a tunable S-curve. More bands → higher recall, lower precision. *Why it matters:* converts the all-pairs O(N²) comparison into a bucketed near-linear pass; the band count is the precision/recall dial you must justify.

**(3) Quality-classifier training signal.** Positives = text from trusted-source-linked pages (Wikipedia external-reference URLs, the GPT-2/WebText "linked-by-Reddit-karma" idea); negatives = random CommonCrawl. Train fastText; the score thresholds precision vs recall of "keep." *Why it matters:* "quality" is *defined operationally by the label source* — there is no ground truth. The reasoningLLM analogue: what is the trusted-positive source for *reward-data* quality? (see §3, §8).

**(4) n-gram / exact-match contamination detection.** A test item is contaminated if a sufficiently long n-gram (or exact line) appears in training. Same hash-set/Jaccard primitive as dedup, run **across the train/eval boundary** instead of within-corpus. *Why it matters:* this is A4.2 — the load-bearing eval-hygiene gate. An uncaught overlap inflates `true_quality` (the held-out oracle measures memorization, not reasoning), silently shrinking the measured `true_quality_gap` and making VERA dishonest.

**(5) Language-ID + the (label, score) + threshold pattern.** fastText returns `(lang, confidence)`; a filter keeps docs above a confidence threshold. *Why it matters:* every classifier filter in A4 (lang, NSFW, toxic, quality) shares this `(label, score)` interface — a single threshold-sweep harness covers all of them, and is the same shape as the reward-model confidence used downstream.

**(6) Filtering pipeline order.** Cheap/destructive-early vs expensive-late: extract → language-ID (drop non-target early, cheapest signal) → Gopher heuristics (cheap structural) → harmful/NSFW (classifier) → PII mask (transform, not drop) → quality classifier (most expensive) → **dedup last** (operates on the survivors; dedup before filtering wastes work and corrupts frequency counts). *Why it matters:* order changes both cost and the final distribution; per-filter discard accounting (`filter_data` deliverable) is the audit trail.

---

## 3. Map to reasoningLLM_scratch source files

The reframe (per UNIFIED_FRONTIER_PROJECT_SPEC §3, L4·A4): keep the **dedup + contamination machinery**, repoint it at **reward/verifier data**, and turn "clean vs noisy reward data" into a controlled lever on `hack_rate`.

| CS336 deliverable | → reasoningLLM source | Keep/adapt vs course-only |
|---|---|---|
| `minhash_deduplication` (MinHash+LSH, Jaccard, n-gram normalization) | `data/curation.py` | **KEEP — core.** Dedup the RLVR task set so near-duplicate problems don't dominate the rollout grid (biases `hack_rate` estimates). |
| `exact_deduplication` (hash-count line dedup) | `data/curation.py` | **KEEP.** Exact-dup detector for verifier/reward records. |
| `gopher_quality_filters` (heuristic length/symbol/word-char rules) | `data/curation.py` | **ADAPT.** Cheap structural filters for reward-data text (e.g., degenerate/empty rationales, broken verifier transcripts). |
| `quality_classifier` (trusted-positive vs random-negative fastText) | `data/curation.py` | **ADAPT the *signal design*, not the CC scale.** The A4.1 ablation: build **clean-vs-noisy RLVR task sets** and measure whether reward-data quality moves `hack_rate` at fixed compute. The "positive source" question becomes "what is trusted reward-data?" (see §8). |
| n-gram / exact-match overlap (reuse of §3 Jaccard core, run train↔eval) | `data/contamination.py` | **KEEP — load-bearing.** n-gram/exact-match contamination check on VERA eval sets (MATH-500, AIME held-out) so the `true_quality` oracle measures reasoning, not leaked memorization. |
| `language_identification` / `mask_pii` / `harmful_content` | (not on critical path) | **COURSE-ONLY.** Build to pass tests; not repointed at reasoningLLM v0.1.0. |
| `filter_data` full CC-scale leaderboard run | — | **SKIP at scale** (§6). Keep only the pipeline-order + discard-accounting discipline as the template for the curation pipeline's logging. |

**The reframe in one sentence:** clean-vs-noisy reward/verifier data is an *independent variable* — does cleaning it (dedup + quality-filter the reward set) move `hack_rate` at fixed compute? That delta is the A4.1 artifact and it plugs straight into the repo's one number, `true_quality_gap = reward − true_quality (+ hack_rate, kl_train_infer)`.

---

## 4. Map to core context docs

- **UNIFIED_FRONTIER_PROJECT_SPEC §3 → "L4 · A4 — Data: *the underpriced bottleneck*":** Core = CommonCrawl→filter→dedup→quality-classify→train. **Add-on A4.1** = reward/verifier-data curation + ablation: *"Build clean-vs-noisy RLVR task sets; measure whether reward-data quality moves `hack_rate` at fixed compute."* Artifact = a reward-data curation pipeline + the clean-vs-noisy `hack_rate` delta. **Add-on A4.2** = *"contamination/dedup as eval hygiene (Phase 3 discipline)"*: n-gram/exact-match train↔eval check on VERA eval sets; *"Without this, every VERA number is suspect — so it is load-bearing, not optional."* Feeds: env/task curation + the `true_quality` oracle's held-out integrity.
- **UNIFIED_FRONTIER_PROJECT_SPEC §5** (evidence ledger): L4 row = "reward-data curation + contamination check → env/task curation, oracle integrity"; bridges RL/Post-Training RE and Evals/Agents RE JDs; **Phase 3 "become one with the data" — A4's contamination check is non-negotiable before any VERA number.**
- **CAPSTONE_AND_STUDY_PLAN §3, row A4** (line ~129): "A4 — Data (CommonCrawl→filter→dedup→train) | Data quality as a lever | Clean vs noisy **verifier/reward data**: does reward-data quality change `hack_rate`? Also R2E task curation | CS336 L13, L14." Threading order (line ~134): A1→A2→A5→**A3+A4** (the analytical axes)→VERA write-up. Sources (lines ~205–207): data-constrained scaling (arXiv 2305.16264), DataComp-LM (arXiv 2406.11794).
- **STUDY_PLAN_2026 §0.8 (Day 10)** (lines ~246, ~264, ~283): "A4 data filter/dedup | data quality as a lever | clean vs noisy verifier-reward data → `hack_rate`; R2E curation"; **Day-10 ship target = `envs/r2e_wrapper.py`** (env/task curation, advances L5); JD language: "data pipelines" (OpenAI), "curate post-training data" (MSL).
- **repo CLAUDE.md / README, L4 row:** "L4 Data | A4 | reward/verifier-data curation · dedup · contamination | `data/curation.py`, `data/contamination.py`." Each stub's docstring must name its §3 brief, a falsifiable prediction, and a kill criterion.

**Note on A4.2 as a Phase-3 gate:** the contamination check is not a feature, it is a **precondition**. It runs in Phase 3 ("become one with the data") and **gates every VERA number** — no `true_quality_gap` is reported on an eval set that hasn't passed the train↔eval n-gram check first.

---

## 5. The frontier 2026 lens

**Commoditized (do not over-invest):** HTML→text extraction, language-ID, PII regex, NSFW/toxic classifier-apply, basic Gopher heuristics. These are library calls and well-trodden; mastery here is table-stakes, not differentiating. Full CommonCrawl-scale processing is an infra exercise, not a research one.

**Scarce / underpriced (where the leverage is):** *reward/preference-data curation.* "Data, not compute, is the bottleneck" is the quiet 2026 consensus (UNIFIED_FRONTIER_PROJECT_SPEC §3). The least-curated dataset in any RL lab is the reward/verifier data — and it is the one most likely to teach the model to hack. Quality-aware and **data-constrained scaling laws** (Muennighoff et al., arXiv 2305.16264 — repeated-token value decay) and **DataComp-LM** (arXiv 2406.11794 — the benchmark that proved *filtering recipe* beats *more tokens*) are the 2026 signals that data quality changes the scaling *exponent*, not just the offset. **Contamination/eval-hygiene** is the second scarce skill: as verifiers become products, eval methodology *is* the deliverable.

**Frontier JD language this maps to verbatim:** OpenAI — *"data pipelines for LLM training"*; Meta MSL — *"curate critical post-training data (SFT, RLHF)"*; and the interview prompts *"how do you prevent eval contamination?"* and *"how does data quality change a scaling/RL outcome?"* (UNIFIED_FRONTIER_PROJECT_SPEC §3 interview-leverage row).

---

## 6. Prioritization verdict — what matters / what to skip

**The ~20% that is load-bearing:**
1. **The dedup machinery** — `exact_deduplication` + `minhash_deduplication` (MinHash signatures, LSH banding, n-gram/Jaccard normalization). This is the reusable core of `curation.py` *and* the primitive `contamination.py` is built on.
2. **The contamination check** (A4.2) — n-gram/exact-match train↔eval overlap on VERA eval sets. The single most load-bearing thing in A4: without it every `true_quality_gap` is suspect.
3. **The reward-data quality → `hack_rate` ablation** (A4.1) — clean-vs-noisy RLVR task sets, measure the `hack_rate` delta at fixed compute. The one A4 deliverable that produces an *experimental finding*, not just a passing test.

**Course-rote (build to green, do not extend):** `extract_text`, `language_identification`, `mask_pii`, `harmful_content`, `quality_classifier` at CC scale, `gopher_quality_filters`. Pass the adapters; internalize the (label, score)+threshold pattern; move on.

**SKIP list:**
- Processing more than a **token-budget slice** of CommonCrawl — do NOT run the full 5000-WET / 375 GB `filter_data` leaderboard job. Run a small slice to exercise pipeline order + discard accounting, then stop.
- The **Paloma C4-100-domains 200K-iteration leaderboard training run** — pure infra spend, no reasoningLLM signal. Skip the model-training half of `filter_data`.
- Slurm/`submitit` cluster orchestration — `concurrent.futures` on a slice is enough; the cluster path is course-environment-specific.

---

## 7. Build checklist (ordered, with discipline gates)

1. **Feynman first** — no code without a justifying F-ID (CLAUDE.md). Feynman MinHash+LSH and n-gram contamination before touching `curation.py`/`contamination.py`.
2. **`exact_deduplication`** → green `test_exact_line_deduplication`. Hash-keyed line counts. Loss-at-init analogue: assert a known-duplicate corpus collapses to the expected unique-line count.
3. **`minhash_deduplication`** → green `test_minhash_deduplication`. Verify the `P[match] = Jaccard` property on a hand-built pair before trusting the LSH bucketing. Justify the band count (precision/recall S-curve).
4. **Port dedup → `data/curation.py`**, repointed at a tiny reward/verifier task set. Docstring names §3 brief + falsifiable prediction + kill criterion.
5. **n-gram contamination → `data/contamination.py`.** Reuse the Jaccard/n-gram core across the train↔eval boundary.
6. **GATE (mandatory, before any VERA eval number):** run the contamination check on the VERA eval sets (MATH-500, AIME held-out). **No `true_quality_gap` is reported on an eval set that has not passed this check.** This is the Phase-3 "become one with the data" gate; it is non-negotiable (UNIFIED §5, CAPSTONE Phase 3).
7. **A4.1 ablation — predict before you run:** write the falsifiable number first. **Prediction: noisier reward data → higher `hack_rate` at fixed compute.** Build clean-vs-noisy RLVR task sets; run the grid; log the `hack_rate` delta. If clean data does *not* lower `hack_rate`, that is itself the finding — record it, do not bury it.
8. **Discard accounting** — for the slice you do run, log per-filter % discarded (the `filter_data` discipline) as the curation pipeline's audit trail.
9. **Green-CI commit** — `ruff` + `pyright` + `pytest -m "not gpu"`; message references the F-ID: `data: <imperative> (F-NNN)`.

---

## 8. Open questions / ADR triggers

- **A4.1 reward-data noise model.** Which noise model defines "noisy" reward data — label-flip on the verifier, truncated/degenerate rationales, near-duplicate task injection, or distractor-padding? Each predicts a different `hack_rate` mechanism. → **ADR before the ablation.**
- **Quality-classifier positive source for reward data.** CC uses Wikipedia-linked URLs as trusted positives; what is the analogous "trusted positive" for *reward/verifier* data (human-verified transcripts? oracle-confirmed solutions?)? The signal design is the whole game. → **ADR.**
- **Contamination n-gram size for MATH/AIME.** Math problems are short and templated — too-short an n-gram floods false positives, too-long misses paraphrased leakage. What n (and exact-match-vs-fuzzy) for MATH-500 / AIME held-out? → **ADR; this directly sets the trustworthiness of every VERA number.**
- **Dedup aggressiveness vs task diversity.** Over-dedup of the RLVR task set can erase legitimate problem variants and shrink coverage; what Jaccard threshold balances near-dup removal against task diversity? → revisit after the A4.1 grid.
- **Scope guard:** keep MoE-data and multimodal-data curation as ADR stubs only — never smuggle into v0.1.0 (CLAUDE.md frozen scope).
