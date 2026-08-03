#!/bin/bash
# S3.5 phase 1 — pod-side pipeline: tokenizer → corpus staging → s7 batch de-confound pair
# → P5 five-point LR sweep. Designed for tmux; every step logs and skips work already done.
#
#   bash scripts/run_s35_phase1.sh
#
# Phase 2 (confirmation run, width probe, S3.5 ladder points) is launched SEPARATELY after the
# LR winner is picked from artifacts/p5/ — the winner is a pre-registered decision point, not
# something to fire unattended mid-pipeline.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /root/cs336/scratch_llm)"

PY=.venv/bin/python
DATA=artifacts/s3_scaling_sweep/data
LOG=artifacts/s35_phase1.log
TOK_DIR=artifacts/f12_corpus_ablation/climbmix
TARGET_TOKENS=3950000000  # covers s8 (2.71B) + s35_d14r20 (3.90B, the optional rung)

step() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
run() { # point grid batch lr outdir
  local point=$1 grid=$2 batch=$3 lr=$4 outdir=$5
  step "RUN $point (grid $grid, batch $batch, lr ${lr:-default}) -> $outdir"
  $PY scripts/s3_scaling_sweep.py run --grid "$grid" --points "$point" \
    --data-dir "$DATA" --out-dir "$outdir" \
    --device cuda --bf16 --batch "$batch" --seed 0 ${lr:+--lr "$lr"} 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  step "DONE $point rc=$rc"
  return $rc
}

step "phase 1 starting"

# 1. ClimbMix tokenizer (lost with the old pod; deterministic rebuild — see script docstring)
if [ ! -f "$TOK_DIR/tokenizer.json" ]; then
  step "rebuilding climbmix tokenizer"
  $PY scripts/rebuild_arm_tokenizer.py --arm climbmix 2>&1 | tee -a "$LOG" || {
    step "FATAL: tokenizer rebuild failed"; exit 1; }
else
  step "tokenizer present, skipping rebuild"
fi

# 2. Corpus staging (~4B tokens; the GPU idles here — CPU + network bound)
if [ ! -f "$DATA/tokenizer.json" ]; then
  step "staging corpus (target $TARGET_TOKENS tokens)"
  $PY scripts/stage_s3_corpus.py --corpus climbmix --target-tokens "$TARGET_TOKENS" \
    2>&1 | tee -a "$LOG" || { step "FATAL: staging failed"; exit 1; }
else
  step "corpus staged, skipping"
fi

# 3. s7 batch de-confound pair — SAME fresh tokenizer/data/seed/GPU; only batch differs.
#    (batch-4 is not skippable: the original s7 record was measured with the lost tokenizer,
#    so the verdict is internal to this pair. results go to separate out-dirs because
#    SweepRecord carries no batch field.)
run s7 v1 4 "" artifacts/s3_rerun_pair/batch4 || { step "FATAL: s7 batch4 failed"; exit 1; }
run s7 v1 8 "" artifacts/s3_rerun_pair/batch8 || { step "FATAL: s7 batch8 failed"; exit 1; }

# 4. P5 five-point LR sweep at d12 @ ratio-4 (CERTAINTY_PLAN §6): multipliers of the 3e-3
#    base, full schedule shape, one out-dir per LR so records never mix peaks.
for m in 0.5 0.7 1.0 1.4 2.0; do
  if [ "$m" = "1.0" ]; then lr=""; else lr=$($PY -c "print(3e-3 * $m)"); fi
  run p5_d12r4 v3 8 "$lr" "artifacts/p5/lr$m" || { step "FATAL: P5 lr$m failed"; exit 1; }
done

step "phase 1 complete — pick the LR winner (tie-break: lower LR within 0.003 bpb), then launch phase 2"
