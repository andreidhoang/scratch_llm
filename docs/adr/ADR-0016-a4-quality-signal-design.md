# ADR-0016 — A4 quality classifier: the label source *is* the definition of quality

**Status:** accepted · 2026-07-03 (sprint node W7b, ADR-0014)
**Scope:** `src/scratch_llm/data/quality.py` — the A4 quality classifier (15 pts, the PDF's own
emphasis). The A4 guide §7 names this ADR trigger: the choice of positive source — and the
negative sampling — *is* the modeling decision.

## Context

"Quality" for pretraining data has no ground truth. Every production recipe defines it
operationally: GPT-2/WebText used pages linked from Reddit comments with ≥3 karma; CS336 A4 uses
pages linked from Wikipedia external references; DataComp-LM (2024) showed the filtering recipe
outweighs raw token count. A classifier trained on `trusted-positive vs random-negative` does not
*discover* quality — it **amortizes the label-source judgment** over the whole corpus at
fastText prices.

## Decision

- **Positive signal:** text from trusted-source-linked pages — canonical label **`wiki`**
  (Wikipedia external-reference URLs at acceptance scale; any substitute positive set must be
  named in the training call, because swapping it redefines the corpus).
- **Negative signal:** **random CommonCrawl** — canonical label **`cc`** — drawn from the same
  distribution the filter will run on. Negatives from anywhere else teach the model the domain
  gap, not the quality gap.
- **Model:** fastText supervised (`dim=64, epoch=10, lr=0.5, wordNgrams=2, minCount=1`,
  `thread=1` + fixed `seed` for reproducible training). fastText, not a neural ranker: the
  classifier must be cheap enough to run on *every* surviving document.
- **The dial:** the score threshold on the `wiki` label, applied through the one shared
  `(label, score) + threshold` interface (`filters.classifier_keep`). Pipeline default
  `quality_threshold = 0.5` (argmax semantics — keep what looks more wiki than cc);
  raising it is the *only* sanctioned aggressiveness knob.
- **No bundled model.** `classify_quality` loads `models_dir()/quality_wiki_cc.bin` and raises
  with training instructions when absent — shipping a default model would hide the signal-design
  decision this ADR exists to record.

## Justification

**Predict-before-run (the threshold's effect, stated before fitting):** raising the threshold
discards more documents, raises the kept corpus's mean quality, and shrinks it — precision up,
recall of "keep" down. The hermetic test pins the mechanism: keep-fraction is monotone
non-increasing in the threshold (`tests/test_data_quality.py::
test_threshold_keep_fraction_monotone_non_increasing`, on a synthetic pos/neg/mixed eval set).

The pipeline runs quality **after** the cheap destructive filters (langid, Gopher, harmful) so
the most expensive classifier sees the fewest documents, and **before** dedup so signatures are
computed only for text that can actually enter the corpus (guide §2(6); ADR-0015 for the dedup
dials).

## Consequences

- Changing the positive source (e.g. Reddit-karma links, curated domain lists) is a *corpus
  redefinition* → new ADR entry, not a hyperparameter tweak.
- W9 acceptance (`test_classify_quality`: `wiki` on the wiki-reference fixture, `cc` on the CC
  fixture) requires training on scraped wiki-reference positives vs WET negatives and persisting
  to `models_dir()/quality_wiki_cc.bin`; the machinery (`train_quality_classifier(...).save(...)`)
  is complete and tested on synthetic corpora.
- Known trap recorded: fasttext 0.9.3 `train_supervised(wordNgrams=1)` reliably NaNs on
  tiny-vocab corpora; the default `wordNgrams=2` is stable (see the note in
  `tests/test_data_quality.py`).
