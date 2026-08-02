#!/bin/bash
# Run the full S3 scaling sweep (s1-s7) on the operator-selected ClimbMix corpus.
# Designed for tmux: resilient to SSH detach, logs everything, runs until completion.
# OOM handling: tries batch=8, falls back to 4, then 2, logging each deviation.
set -euo pipefail

cd /workspace/scratch_llm
LOG=artifacts/s3_scaling_sweep/s3_sweep.log
mkdir -p artifacts/s3_scaling_sweep

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] S3 sweep wrapper starting" | tee "$LOG"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Protocol deviations: ClimbMix selected by operator override; batch sizes may be reduced for OOM; S3 grid fixed to include qk_norm params." | tee -a "$LOG"

# Wait for ClimbMix staging to finish (if still running)
echo "Checking for active staging process..." | tee -a "$LOG"
while pgrep -f "stage_s3_corpus.py --corpus climbmix" >/dev/null; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] staging still running; waiting 30s" | tee -a "$LOG"
    sleep 30
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] staging process exited." | tee -a "$LOG"

# Confirm staged data exists
if [ ! -f artifacts/s3_scaling_sweep/data/tokenizer.json ]; then
    echo "ERROR: staged tokenizer not found at artifacts/s3_scaling_sweep/data/tokenizer.json" | tee -a "$LOG"
    exit 1
fi

# Per-point batch map. s1-s3 are small (d=4, 20M); s4-s6 are d=8, 59M; s7 is d=12, 135M.
# F12 (d=6, 35.8M, batch=8) used ~17 GB on RTX 5090 32 GB.
# s4-s6 estimated ~25-30 GB -> batch=8 is tight but likely OK.
# s7 estimated >32 GB at batch=8 -> start with 4.
declare -A BATCH_MAP=(
    [s1]=8 [s2]=8 [s3]=8
    [s4]=8 [s5]=8 [s6]=8
    [s7]=4
)

run_point() {
    local point=$1
    local batch=$2
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running $point with batch=$batch" | tee -a "$LOG"
    uv run python scripts/s3_scaling_sweep.py run \
        --points "$point" \
        --data-dir artifacts/s3_scaling_sweep/data \
        --out-dir artifacts/s3_scaling_sweep \
        --device cuda \
        --bf16 \
        --batch "$batch" \
        --seed 0 2>&1 | tee -a "$LOG"
}

for point in s1 s2 s3 s4 s5 s6 s7; do
    batch=${BATCH_MAP[$point]}
    if run_point "$point" "$batch"; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $point completed with batch=$batch" | tee -a "$LOG"
        continue
    fi

    # OOM / failure fallback for s7: try batch=4 -> 2
    if [ "$point" = "s7" ] && [ "$batch" -eq 4 ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $point failed at batch=4; retrying batch=2" | tee -a "$LOG"
        run_point "$point" 2
    else
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $point failed at batch=$batch; aborting sweep" | tee -a "$LOG"
        exit 1
    fi
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] S3 sweep run complete; fitting scaling law" | tee -a "$LOG"

uv run python scripts/s3_scaling_sweep.py fit \
    --out-dir artifacts/s3_scaling_sweep 2>&1 | tee -a "$LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] S3 fit complete" | tee -a "$LOG"
