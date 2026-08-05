#!/bin/bash
# S3.5 phase 2 — η* confirmation, width probe, N-ladder rungs.
# Runs the 3 CHEAP points first (~20h, ~$11), then HALTS at a gate.
# The expensive s35_d14r20 (~33h, ~$17.6) must be launched explicitly:
#   touch artifacts/p5_phase2_proceed_d14r20   then re-run this script.
#
# η* = 0.0021 (phase-1 P5 winner, val_bpb 0.4033). Pre-registered: held fixed
# across the whole ladder (de-confounding requirement). Batch 8, ctx 2048, seed 0.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /root/cs336/scratch_llm)"

PY=.venv/bin/python
DATA=artifacts/s3_scaling_sweep/data
LOG=artifacts/s35_phase2.log
ETA_STAR=0.0021   # phase-1 LR winner — held FIXED (do not re-tune)

step() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
run() { # point grid batch lr outdir
  local point=$1 grid=$2 batch=$3 lr=$4 outdir=$5
  if [ -s "$outdir/results.json" ]; then step "SKIP $point ($outdir/results.json exists)"; return 0; fi
  step "RUN $point (grid $grid, batch $batch, lr ${lr:-default}) -> $outdir"
  $PY scripts/s3_scaling_sweep.py run --grid "$grid" --points "$point" \
    --data-dir "$DATA" --out-dir "$outdir" \
    --device cuda --bf16 --batch "$batch" --seed 0 ${lr:+--lr "$lr"} 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  step "DONE $point rc=$rc"
  return $rc
}

step "=== phase 2 starting (eta*=$ETA_STAR held fixed) ==="

# Tier A — cheap validation (total ~20h, ~$11)
# 1. Width probe: p5_d8r4 — does η* transfer DOWN the width axis? (~0.6h)
#    Expected: clean convergence. If loss spikes/d diverges, η* is width-dependent -> STOP.
run p5_d8r4 v3 8 "$ETA_STAR" artifacts/p5_phase2/d8r4_widthprobe \
  || { step "FATAL: d8r4 width probe failed — η* may not transfer; HALT before more spend"; exit 1; }

# 2. Confirmation: s7 @ ratio-8 with η* (~6.3h). Compare to phase-1 s7 batch8 (0.3864, default LR).
#    Pre-registered gate: η* must reach val_bpb within 0.003 of 0.3864. If WORSE by >0.003,
#    the default LR was near-optimal at r8 and η* doesn't transfer across ratios -> STOP.
run s7 v1 8 "$ETA_STAR" artifacts/p5_phase2/s7_confirm_eta \
  || { step "FATAL: s7 confirm failed"; exit 1; }

# 3. N-ladder rung: s35_d14r8 (~13.1h). First point at a NEW depth — anchors the α direction.
run s35_d14r8 v3 8 "$ETA_STAR" artifacts/p5_phase2/d14r8 \
  || { step "FATAL: s35_d14r8 failed"; exit 1; }

step "=== TIER A COMPLETE (3 cheap points done) — GATE CHECK ==="
step "Gate criteria (see script docstring):"
step "  d8r4 width probe:  did it converge cleanly? (val_bpb should be > d12r4=0.4033 since smaller model)"
step "  s7 confirm eta*:   within 0.003 of phase-1 s7 batch8=0.3864?"
step "  d14r8 ladder:      val_bpb < d12r4=0.4033? (bigger model should be lower)"
step "Inspect: for f in artifacts/p5_phase2/*/results.json; do echo \$f; cat \$f; done"
step "To proceed to the expensive d14r20 (~33h, ~\$17.6):"
step "  touch artifacts/p5_phase2_proceed_d14r20 && bash scripts/run_s35_phase2.sh"
step "Otherwise this is the natural stop point. Pod can be destroyed after committing results."

# Tier B — expensive (~33h, ~$17.6). Only runs if the gate file exists.
if [ -f artifacts/p5_phase2_proceed_d14r20 ]; then
  step "Gate file present — proceeding to s35_d14r20 (THE BIG RUN)"
  run s35_d14r20 v3 8 "$ETA_STAR" artifacts/p5_phase2/d14r20 \
    || { step "FATAL: s35_d14r20 failed"; exit 1; }
  step "=== PHASE 2 FULLY COMPLETE — all 4 points done ==="
else
  step "Gate file absent — stopping after tier A. Results ready for analysis."
fi
