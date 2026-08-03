#!/bin/bash
# S3.5 phase 2 — launched AFTER the P5 winner is picked from artifacts/p5/ (the pre-registered
# decision point: min val_bpb, tie-break = lower LR within 0.003 bpb).
#
#   bash scripts/run_s35_phase2.sh WINNER_LR [--with-d14r20]
#
# Sequence (CERTAINTY_PLAN §6 + SCALING_PROGRAM stage 2), all at recipe-v1 = WINNER_LR:
#   1. confirmation: d12 @ ratio-8 (= v1 s7) at WINNER_LR — doubles as the T2 anchor
#   2. width probe: d8 @ ratio-4 at {0.7, 1.0, 1.4} × WINNER_LR
#   3. ladder refresh: s1–s6 at WINNER_LR — MANDATORY for the joint fit. The rebuilt
#      tokenizer is not verifiably identical to the lost original, so the banked s1–s6
#      records (old tokenizer, old recipe) cannot be mixed into the S3.5 fit; without
#      these six cheap points the 5-parameter joint law is underdetermined.
#   4. S3.5 ladder: s8 (d12 @ ratio-20), s35_d14r8; s35_d14r20 only with --with-d14r20
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /root/cs336/scratch_llm)"

WINNER="${1:?usage: run_s35_phase2.sh WINNER_LR [--with-d14r20]}"
WITH_D14R20="${2:-}"

PY=.venv/bin/python
DATA=artifacts/s3_scaling_sweep/data
LOG=artifacts/s35_phase2.log

step() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
run() { # point grid batch lr outdir
  local point=$1 grid=$2 batch=$3 lr=$4 outdir=$5
  step "RUN $point (grid $grid, batch $batch, lr $lr) -> $outdir"
  $PY scripts/s3_scaling_sweep.py run --grid "$grid" --points "$point" \
    --data-dir "$DATA" --out-dir "$outdir" \
    --device cuda --bf16 --batch "$batch" --seed 0 --lr "$lr" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  step "DONE $point rc=$rc"
  return $rc
}

step "phase 2 starting — recipe-v1 peak LR = $WINNER"

# 1. Confirmation run (d12 @ ratio-8 at the winner) — T2 anchor
run s7 v1 8 "$WINNER" artifacts/p5/confirm_r8 || { step "FATAL: confirm failed"; exit 1; }

# 2. Width probe at d8 @ ratio-4: does the LR optimum drift with width?
for m in 0.7 1.0 1.4; do
  lr=$($PY -c "print($WINNER * $m)")
  run p5_d8r4 v3 8 "$lr" "artifacts/p5/width_lr$m" || { step "FATAL: width probe $m failed"; exit 1; }
done

# 3. Ladder refresh at recipe-v1 (cheap points first — fail fast if the recipe misbehaves)
for p in s1 s2 s3 s4 s5 s6; do
  run "$p" v1 8 "$WINNER" "artifacts/s35/ladder_$p" || { step "FATAL: ladder $p failed"; exit 1; }
done

# 4. S3.5 top-end points (recipe-v1)
run s8 v1 8 "$WINNER" artifacts/s35/d12_r20 || { step "FATAL: s8 failed"; exit 1; }
run s35_d14r8 v3 8 "$WINNER" artifacts/s35/d14_r8 || { step "FATAL: d14r8 failed"; exit 1; }
if [ "$WITH_D14R20" = "--with-d14r20" ]; then
  run s35_d14r20 v3 8 "$WINNER" artifacts/s35/d14_r20 || { step "FATAL: d14r20 failed"; exit 1; }
fi

step "phase 2 complete — sync artifacts back, then fit + gates + charts on the laptop"
