"""A4 data — the CommonCrawl extract → filter → quality-classify → exact + MinHash/LSH dedup
pipeline (to build); pipeline order + per-filter discard accounting are the load-bearing 20%.

Plus the pretraining shard substrate (A1, close-the-loop front): ``shards.py`` — docs →
memmap uint16 token shards that ``train.py::get_batch`` consumes. Heavy A4 deps stay lazy
(import the submodules directly); the shard layer is numpy-only, so it exports here.

Plus the A0 decontamination gate: ``decontaminate.py`` — n=13 exact-collision word n-grams
against the eval sets, threaded into shard building via the ``doc_filter`` seam (stdlib-only
at import time, so it exports here too)."""

from scratch_llm.data.decontaminate import (
    DecontamResult,
    build_eval_ngrams,
    decontam_doc_filter,
    decontaminate_docs,
    default_eval_texts,
    ngram_overlap,
)
from scratch_llm.data.shards import (
    DOC_SEPARATOR,
    ShardMeta,
    build_dataset,
    download_fineweb_slice,
    load_dataset_tokens,
    load_shard,
    load_tokenizer,
    tokenize_to_shard,
)

__all__ = [
    "DOC_SEPARATOR",
    "DecontamResult",
    "ShardMeta",
    "build_dataset",
    "build_eval_ngrams",
    "decontam_doc_filter",
    "decontaminate_docs",
    "default_eval_texts",
    "download_fineweb_slice",
    "load_dataset_tokens",
    "load_shard",
    "load_tokenizer",
    "ngram_overlap",
    "tokenize_to_shard",
]
