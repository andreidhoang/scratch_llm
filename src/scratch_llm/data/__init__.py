"""A4 data — the CommonCrawl extract → filter → quality-classify → exact + MinHash/LSH dedup
pipeline (to build); pipeline order + per-filter discard accounting are the load-bearing 20%.

Plus the pretraining shard substrate (A1, close-the-loop front): ``shards.py`` — docs →
memmap uint16 token shards that ``train.py::get_batch`` consumes. Heavy A4 deps stay lazy
(import the submodules directly); the shard layer is numpy-only, so it exports here."""

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
    "ShardMeta",
    "build_dataset",
    "download_fineweb_slice",
    "load_dataset_tokens",
    "load_shard",
    "load_tokenizer",
    "tokenize_to_shard",
]
