# Runbook — A4 `filter_data` pipeline slice (real WET, real fastText models)

> ## 🟢 RUN-ON-CPU — not blocked, no GPU needed
> This is the one A4 deliverable you actually *run*. It exercises the canonical CommonCrawl
> curation pipeline **end-to-end on real Common Crawl WET files with the real fastText models**,
> and prints the per-filter discard accounting that *is* the `filter_data` written deliverable.
> It is **CPU-only** — run it **on THIS standing box** (128 vCPU / 251 GB RAM, everything already
> installed) for **$0**, or on a **cheap CPU rental** if the standing box is busy with perf/GPU
> work. The full **5000-WET / 375 GB leaderboard + Paloma C4-100 perplexity run stays SKIP** —
> see [§0](#0-scope-what-you-run-vs-what-stays-skip).
>
> A fresh agent on a fresh pod can execute this top-to-bottom with no other context.

---

## 0. Scope: what you run vs what stays SKIP

| | Run here (LOAD-BEARING) | SKIP (capped infra sink) |
|---|---|---|
| **Data** | a **token-budget slice** (~2–5 M words, 1–3 raw WET files) | the full **5000-WET / 375 GB** English crawl |
| **What it proves** | pipeline **order** + **per-filter discard accounting** + the "web data is mostly garbage" lesson | a Paloma C4-100 perplexity leaderboard rank |
| **Compute** | minutes of CPU | 200K-iter GPT-2 training on a GPU cluster |

**Why the full run is SKIP** (from `docs/assignment_guides/A4_data_BUILD_GUIDE.md` §5–6 and
`docs/EXECUTION_SPEC_CS336_FINISH.md` W10): the 5000-WET/375 GB `filter_data` run + the
`tokenize_data` → `train_model` (200K-iter GPT-2, Paloma C4-100) leaderboard are pure
GPU-dollar/infra spend with **no mastery carry**. The load-bearing 20% — the pipeline order, the
discard audit trail, and reading the data — is fully captured by a small slice. Run the slice,
log the discard table, stop.

## 1. Purpose + which `scratch_llm` code this exercises (real paths)

Turn raw web HTML-derived text into a trainable corpus through the one ordering that survives
scrutiny, charging every dropped document to the first filter that killed it:

```
extract → language-ID → Gopher → NSFW → toxic → PII-mask (transform) → quality → dedup LAST
```

Code under test (all real, all already implemented + green on this box):

| File | What the slice drives |
|---|---|
| `src/scratch_llm/data/pipeline.py` | `filter_data()`, `FilterConfig`, `FilterConfig.with_default_models()`, `FilterReport.discard_table()`, `STAGE_ORDER` — the driver + audit trail |
| `src/scratch_llm/data/filters.py` | `identify_language`, `classify_nsfw`, `classify_toxic_speech`, `mask_emails/phones/ips`, `gopher_quality_filter`, `extract_text_from_html_bytes`, `MODEL_URLS`, `models_dir()`, `ensure_model()` |
| `src/scratch_llm/data/quality.py` | `QualityClassifier.train/.save/.classify`, `classify_quality`, `DEFAULT_QUALITY_MODEL_FILENAME` (`quality_wiki_cc.bin`) — the **signal-design** step |
| `src/scratch_llm/data/dedup.py` | `minhash_dedup` (MinHash→LSH→Jaccard-confirm→union-find), `DEFAULT_MINHASH_PARAMS` (k=100, b=10, n=5, J≥0.8) — runs LAST, on survivors only |

Oracle already green (run before you trust the slice):
`pytest -m "not gpu" tests/test_data_filters.py tests/test_data_dedup.py tests/test_data_quality.py`.

> **One honest nuance — WET is *pre-extracted* text.** CommonCrawl **WET** = "WARC-Extracted-Text":
> each `conversion` record is already plain text, so the pipeline's **`extract` stage is a
> pass-through** on WET (it only drops empty/whitespace-only records). That is fine — the WET slice
> is what A4 §2.1/§4 asks for. To *also* exercise the real `resiliparse` HTML→text extractor
> (`extract_text_from_html_bytes`), feed raw **WARC** `response` records (HTML bytes) instead — the
> variant is noted in [§7](#7-optional-exercise-the-resiliparse-extractor-on-warc). The language cut
> is the real workhorse of the WET slice.

## 2. Where to run — decision

**Default: run on THIS box.** Everything below is already true here — verified 2026-07-04:

- venv `/workspace/scratch_llm/.venv` (Python **3.12.13**, torch 2.12.1+cu130, sm120 — GPU unused).
- `[data]` extras present: `fasttext`, `resiliparse`, `fastwarc`, plus `scipy`/`matplotlib`.
- **All three fastText models already on disk** at `/workspace/.data_models/`
  (`lid.176.bin` 131 MB, `jigsaw_fasttext_bigrams_nsfw_final.bin` 992 MB,
  `jigsaw_fasttext_bigrams_hatespeech_final.bin` 992 MB) — **[§4](#4-model-files) is a no-op here.**
- Official A4 fixtures for training the quality classifier at
  `/workspace/lectures/assignment4-data/tests/fixtures/` (`high_quality_wiki_reference.txt`,
  `low_quality_cc.txt`).
- 128 vCPU / 251 GB RAM / 43 GB free on `/workspace` — a 1–3 WET slice fits easily.

If the standing box is saturated by the perf/GPU track, rent the **cheapest CPU-heavy box** — this
job never touches CUDA, so **GPU tier is irrelevant; optimize for vCPU count + $/hr, not GPU**.
Vast.ai only rents GPU hosts, so pick the lowest-cost offer with many cores and ignore the GPU:

```bash
# Cheapest host with lots of CPU + enough RAM/disk; GPU is along for the ride, unused.
vastai search offers 'cpu_cores>=16 cpu_ram>=32 disk_space>=60 inet_down>=200 \
  reliability>0.97 rentable=true verified=true' -o 'dph_total' --raw | jq -r '.[0] | {id,dph_total,cpu_cores,cpu_ram,gpu_name}'

OFFER_ID=<id from above>
vastai create instance "$OFFER_ID" --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
  --disk 60 --ssh --direct --raw            # ~$0.05–0.30/hr; see §9 cost
```
(This mirrors `deploy/01_launch.sh`, minus the `gpu_name=` filter. Offer IDs are never hardcoded —
query the live market each run.) Then bootstrap ([§3](#3-fresh-pod-bootstrap)).

## 3. Fresh-pod bootstrap

Skip this entire section on the standing box (already done). On a fresh rental:

```bash
# 1. Clone + one-command env (uv venv, [dev], torch, hooks, memory). See docs/VASTAI_BOOTSTRAP.md.
git clone https://github.com/andreidhoang/scratch_llm && cd scratch_llm
bash scripts/bootstrap-pod.sh

# 2. This job needs the A4 [data] extras (bootstrap installs [dev] core only):
uv pip install --python .venv/bin/python -e ".[data,scaling]"

source .venv/bin/activate
python -c "import fasttext, resiliparse, fastwarc; print('data extras OK')"
```

The official A4 fixtures (quality-classifier training text) ship inside the scaffold checkout at
`/workspace/lectures/assignment4-data/tests/fixtures/`. On a fresh rental where `../lectures/`
is absent, clone it (`git clone https://github.com/stanford-cs336/assignment4-data \
/workspace/lectures/assignment4-data`) **or** supply your own trusted-positive text in §5.

Make a scratch working dir for everything below (nothing here is committed to the repo):

```bash
export SLICE=/workspace/a4_slice && mkdir -p "$SLICE"/{wet,out}
```

## 4. Model files

Three fastText models drive language-ID + the two harmful classifiers. They **auto-download on
first use** via `filters.ensure_model()` (atomic `.partial` → rename), or fetch them explicitly.
The canonical URLs (from `filters.MODEL_URLS`):

| Model | File (`models_dir()` = `$WORKSPACE/.data_models`) | URL | Size |
|---|---|---|---|
| Language ID (lid.176) | `lid.176.bin` | `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin` | 131 MB |
| NSFW (Dolma/Jigsaw) | `jigsaw_fasttext_bigrams_nsfw_final.bin` | `https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin` | 992 MB |
| Toxic/hate (Dolma/Jigsaw) | `jigsaw_fasttext_bigrams_hatespeech_final.bin` | `https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin` | 992 MB |

```bash
# On a fresh box, pre-fetch all three (skips instantly if already present, e.g. THIS box):
python - <<'PY'
from scratch_llm.data import filters
for m in (filters.LID_MODEL, filters.NSFW_MODEL, filters.TOXIC_MODEL):
    print("ready:", filters.ensure_model(m))
PY
```

The **quality** classifier (`quality_wiki_cc.bin`) is *deliberately not bundled* — you train it in
§5, because *choosing the positive label source is the modeling decision* (ADR-0016).

## 5. Train the quality classifier (the signal-design step, ADR-0016)

"Quality" has no ground truth — it is **defined operationally by the label source**. The A4 recipe:
**positives = trusted-source-linked text** (Wikipedia external-reference pages; here the
`high_quality_wiki_reference.txt` fixture), **negatives = random Common Crawl** (drawn from the
*same* WET slice you will filter — so the classifier learns the quality gap, not a domain gap).

This is folded into the driver in §6 (it samples negatives from the freshly downloaded WET), so no
separate step is required. The relevant call it makes:

```python
from scratch_llm.data.quality import QualityClassifier
qc = QualityClassifier.train(pos_texts, neg_texts)      # fastText, single-threaded, seed=0 → reproducible
qc.save(models_dir() / "quality_wiki_cc.bin")           # persisted default consumed by classify_quality
```

**Predict before you train** (write it down first): raising the keep threshold (`quality_threshold`,
default 0.5) → *more* documents discarded, *higher* mean quality, *smaller* corpus (monotone). If
the trained classifier keeps ~100% or ~0% of the slice, the labels are degenerate — see kill
criteria (§8).

## 6. Acquire the WET slice + run the pipeline

### 6a. Pick a crawl + download a few raw WET files

Common Crawl is reachable from this box. Newest crawls (from `collinfo.json`, verified 2026-07-04):
`CC-MAIN-2026-25`, `-2026-21`, `-2026-17` (the scaffold's default), … Use a recent one:

```bash
export CRAWL_ID=CC-MAIN-2026-25          # or any id from https://index.commoncrawl.org/collinfo.json
export N_WET=2                            # 1–3 raw WET files ≈ enough for a 2–5 M-word slice
```

Download `wet.paths.gz`, take the first `N_WET` file URLs, fetch them (each ~120–170 MB gz,
~350–450 MB decompressed, ~30–50 k `conversion` records):

```bash
python - <<'PY'
import gzip, os, urllib.request
from pathlib import Path
BASE = "https://data.commoncrawl.org/"
crawl, n, dst = os.environ["CRAWL_ID"], int(os.environ["N_WET"]), Path(os.environ["SLICE"]) / "wet"
paths_gz = dst / "wet.paths.gz"
if not paths_gz.exists():
    urllib.request.urlretrieve(f"{BASE}crawl-data/{crawl}/wet.paths.gz", paths_gz)
urls = [BASE + line for line in gzip.decompress(paths_gz.read_bytes()).decode().splitlines()[:n]]
for u in urls:
    out = dst / u.split("/")[-1]
    if not out.exists():
        print("downloading", u); urllib.request.urlretrieve(u, out)
    print("have", out, f"{out.stat().st_size/1e6:.0f} MB")
PY
```

> These are **raw, multilingual** WET files (not the scaffold's pre-English-filtered output) — that
> is deliberate: it makes the **language-ID stage do real work** (the dominant cut), which is the
> point of running the slice.

### 6b. The slice driver

Write this driver to the scratch dir and run it. It reads WET `conversion` records up to a token
budget, trains the quality classifier (wiki-positive vs random-CC-negative from the slice), runs
`filter_data`, prints the discard table, and dumps annotation samples for §8. Every line is grounded
in the real `scratch_llm.data` API (verified against `tests/test_data_filters.py`).

```bash
cat > "$SLICE/run_slice.py" <<'PY'
"""A4 filter_data slice — real WET → scratch_llm.data.pipeline.filter_data → discard accounting."""
import os, random
from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from scratch_llm.data.pipeline import filter_data, FilterConfig
from scratch_llm.data.quality import QualityClassifier
from scratch_llm.data.filters import models_dir

SLICE = Path(os.environ["SLICE"])
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "3_000_000"))   # ~words; the "token-budget slice"
FIXTURES = Path("/workspace/lectures/assignment4-data/tests/fixtures")
rng = random.Random(336)

# 1. Read WET conversion records (already plain text) up to the token budget.
docs, ids, tokens = [], [], 0
for wet in sorted(SLICE.glob("wet/*.warc.wet.gz")):
    for rec in ArchiveIterator(str(wet), record_types=WarcRecordType.conversion):
        text = rec.reader.read().decode("utf-8", errors="replace")
        if not text.strip():
            continue
        docs.append(text)
        ids.append(rec.headers.get("WARC-Target-URI") or f"doc{len(ids):06d}")
        tokens += len(text.split())
        if tokens >= TOKEN_BUDGET:
            break
    if tokens >= TOKEN_BUDGET:
        break
# doc_ids must be unique (pipeline asserts this) — dedup URIs by suffixing collisions.
seen = {}
uids = []
for u in ids:
    seen[u] = seen.get(u, -1) + 1
    uids.append(u if seen[u] == 0 else f"{u}#{seen[u]}")
print(f"slice: {len(docs)} docs, ~{tokens:,} words from {os.environ.get('N_WET','?')} WET file(s)")

# 2. Signal design (ADR-0016): positives = trusted wiki-ref text; negatives = random CC from THIS slice.
pos = [l for l in (FIXTURES / "high_quality_wiki_reference.txt").read_text().splitlines() if l.strip()]
neg = [" ".join(d.split()) for d in rng.sample(docs, min(len(docs), max(50, len(pos))))]
qc = QualityClassifier.train(pos, neg)                          # fastText, seed=0
qc.save(models_dir() / "quality_wiki_cc.bin")
print(f"quality classifier trained: {len(pos)} wiki-pos vs {len(neg)} random-CC-neg")

# 3. Run the canonical pipeline with ALL real classifiers wired.
config = FilterConfig.with_default_models(quality_classifier=qc.classify)
report = filter_data(docs, config, doc_ids=uids, work_dir=SLICE / "out")

# 4. The deliverable: per-filter discard accounting.
print("\n" + report.discard_table())
assert report.total_in == report.kept + sum(report.discarded.values()), "ACCOUNTING BROKEN — stop"

# 5. Annotation dumps (§8): 25 raw docs (look_at_cc) + 5 kept / 5 removed (inspect_filtered_data).
(SLICE / "annotate").mkdir(exist_ok=True)
(SLICE / "annotate/look_at_cc_25raw.txt").write_text(
    "\n\n===RECORD===\n\n".join(f"[{uids[i]}]\n{docs[i][:1500]}" for i in range(min(25, len(docs)))))
kept_ids = {d.doc_id for d in report.survivors}
kept = [d.text for d in report.survivors[:5]]
removed = [docs[i] for i, u in enumerate(uids) if u not in kept_ids][:5]
(SLICE / "annotate/inspect_5kept_5removed.txt").write_text(
    "##### 5 KEPT #####\n\n" + "\n\n---\n\n".join(t[:1500] for t in kept) +
    "\n\n##### 5 REMOVED #####\n\n" + "\n\n---\n\n".join(t[:1500] for t in removed))
print("\nannotation files → ", SLICE / "annotate")
PY

TOKEN_BUDGET=3000000 python "$SLICE/run_slice.py" 2>&1 | tee "$SLICE/out/slice.log"
```

`filter_data` also logs the same discard table at `INFO` on the `scratch_llm.data.pipeline` logger.

## 7. (Optional) Exercise the `resiliparse` extractor on WARC

WET skips real HTML→text extraction (§1). To drive `extract_text_from_html_bytes` for real, swap the
WET download for a **WARC** file from the same crawl (`warc.paths.gz` instead of `wet.paths.gz`),
iterate `WarcRecordType.response` records, and pass the **raw HTML bytes** into `filter_data` (the
pipeline calls `resiliparse` on `bytes` inputs automatically — see `pipeline._extract`). Expect the
`extract` stage to now carry a real 5–15% drop (blank/unparseable pages). This is optional colour;
the WET slice is the graded artifact.

## 8. Become one with the data — the annotation step (`look_at_cc` / `inspect_filtered_data`)

This is the real A4 lesson (§2.1, 4 pts): **web data is mostly garbage**, and aggregate metrics hide
failure modes. Do it by hand — no adapter, no shortcut:

1. **`look_at_cc` — 25 raw records.** Open `$SLICE/annotate/look_at_cc_25raw.txt`. For each of the
   25, jot **language · domain · page-type** (article / product / nav-spam / forum / boilerplate) and
   a one-word quality verdict. Answer the canonical question: **how many records until you hit one
   genuinely high-quality page?** (Expect ~1 in 10–20 — that ratio *is* the lesson.)
2. **`inspect_filtered_data` — 5 kept + 5 removed.** Open
   `$SLICE/annotate/inspect_5kept_5removed.txt`. For each, write one line: *which* filter should have
   caught it and whether you agree. Look hard for **false drops** (a good page killed by the quality
   classifier's register bias, or a legit page erased as a near-dup) — if you find them, that is a
   signal to retune a threshold, not a bug to ignore.

Record the 25-doc tally + the 5/5 justifications in your writeup / `docs/` note — that hand
annotation is the deliverable, even though it is not reusable code.

## 9. Predicted numbers (predict-before-run) + kill criteria

Write these **before** the run; they are the debugging + learning anchor (FOP-2/3). Grounded in
CommonCrawl reality (~40–45% English) + the A4 filter design. Per-stage % is *of documents entering
that stage* (the discard-table convention):

| Stage | Predicted discard (of entering) | Why |
|---|---|---|
| extract | **~0%** (0–3%) | WET is pre-extracted; only empty/whitespace records drop |
| language (`en` ≥0.65) | **~45–60%** | raw WET is multilingual; the big, cheap, early cut |
| gopher | **~15–30%** | nav-spam / too-short / low-alpha boilerplate |
| nsfw | **~1–3%** | small on general crawl |
| toxic | **~1–2%** | small; note the rant-doc books to *nsfw* first (earlier stage) |
| quality (`wiki` ≥0.5) | **~65–90%** | aggressive — most random CC scores `cc`; the dominant late cut |
| dedup (J≥0.8) | **~5–15%** | within-slice near-dups (shared boilerplate) |
| **overall keep** | **~4–9%** | ≈ 0.5·0.78·0.98·0.99·0.25·0.9 → "mostly garbage" confirmed |

Expected wall time: **~2–10 min** single-process for a ~3 M-word / ~15 k-doc slice on this box
(fastText predict dominates; ~hundreds–thousands docs/s). Peak RAM ~2.5 GB (two 992 MB Jigsaw
models resident) + working set — trivial against 251 GB.

**Kill criteria (STOP and diagnose, do not log a number):**
- **Accounting identity fails** (`total_in ≠ kept + Σ discarded`) → pipeline bug. The driver
  `assert`s this; it must never fire.
- **Language keeps >90%** on raw multilingual WET → `lid.176` not loaded / wrong threshold / you
  grabbed pre-filtered English WET.
- **Quality keeps ~100% or ~0%** → degenerate classifier: bad/duplicated labels, wrong `keep_label`
  (must be `"wiki"`), or positives ≈ negatives. Re-check §5 before trusting any discard number.
- **Extract drops >20%** → WET reader mis-parsing (wrong record type / decode) — you are dropping
  real text.
- **Dedup drops >50%** on a single-crawl slice → threshold/normalization too aggressive (erasing
  legitimate variety) — inspect the removed set (§8) before accepting.

If a stage lands far outside its band, the *annotation* (§8) tells you whether the filter is wrong or
your prediction was — that reconciliation is the graded skill.

## 10. Where results go

- **`bench/RESULTS.md`** (main-track section, **append-only**, additive edit — pull-rebase first):
  log one row with the run date, crawl id + slice size (docs / words / N_WET), the **measured
  per-stage discard %** vs the **predicted** band above, overall keep %, wall time, and the one-line
  root cause of the biggest surprise. Mark it `[FACT]` (measured under a fixed seed).
- **The discard table + slice log**: `$SLICE/out/slice.log`.
- **The annotation deliverables**: `$SLICE/annotate/` (the 25-doc tally + 5/5 justifications go into
  your writeup or a `docs/` note; reference them from the RESULTS row).
- **The trained quality classifier**: `$WORKSPACE/.data_models/quality_wiki_cc.bin` (persists on the
  standing box; on a rental it dies with the pod — that is fine, it is cheap to retrain and the
  *signal-design choice*, not the binary, is the artifact).

## 11. Cost estimate

| Where | Compute | Bandwidth | $ |
|---|---|---|---|
| **THIS standing box** | ~2–10 CPU-min | 0 (models + fixtures already local) | **$0** |
| **Cheap CPU rental** | <30 min wall | ~2.1 GB models + ~0.3–0.9 GB WET (2 files) | **< $0.50** (host ~$0.05–0.30/hr; job <0.5 hr) |
| ~~Full 5000-WET + Paloma leaderboard~~ | ~~cluster-days + GPU training~~ | ~~375 GB~~ | **SKIP** — capped infra sink, no mastery carry (§0) |

**Recommendation: run it on the standing box for $0.** Rent only if the box is occupied; when done,
`vastai destroy instance <id>` to stop billing.
