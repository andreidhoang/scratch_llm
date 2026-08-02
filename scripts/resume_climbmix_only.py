#!/usr/bin/env python3
"""Resume only the ClimbMix arm of F12 (FineWeb-EDU result already exists)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.data.shards import DEFAULT_BPE_TRAIN_BYTES, download_fineweb_slice
from scratch_llm.eval.core_suite import core_task_specs
from scratch_llm.eval.corpus_ablation import (
    CorpusAblationArm,
    HeldOutSet,
    _run_arm,
    arm_result_to_dict,
)
from scratch_llm.train import TrainConfig

OUT_DIR = REPO / "artifacts" / "f12_corpus_ablation"
CACHE = OUT_DIR / "parquet"
TARGET_TOKENS = 700_000_000
BATCH = 8
CTX = 2048
STEPS = TARGET_TOKENS // (BATCH * CTX)

held_out = HeldOutSet(docs=tuple(download_fineweb_slice(n_docs=2048, offset=500_000)))

train_cfg = TrainConfig(
    max_steps=STEPS,
    batch_size=BATCH,
    context_length=CTX,
    max_lr=3e-3,
    warmup_steps=max(1, STEPS // 20),
    eval_every=max(1, STEPS // 20),
    optimizer="muon_adamw",
    amp_dtype="bf16",
    device="cuda",
    seed=0,
)

result = _run_arm(
    CorpusAblationArm.CLIMBMIX,
    None,
    held_out,
    OUT_DIR,
    train_cfg,
    depth=6,
    vocab_size=32768,
    max_train_bytes=DEFAULT_BPE_TRAIN_BYTES,
    target_tokens=TARGET_TOKENS,
    decontaminate=True,
    extra_eval_paths=(),
    core_specs=core_task_specs(),
    core_limit=None,
    num_workers=8,
    prebuilt_dir=OUT_DIR / "climbmix",
    progress=print,
)

path = OUT_DIR / "climbmix_result.json"
path.write_text(json.dumps(arm_result_to_dict(result), indent=2), encoding="utf-8")
print(f"\nCLIMBMIX DONE: val_bpb={result.bpb.bits_per_byte:.6f} -> {path}")
