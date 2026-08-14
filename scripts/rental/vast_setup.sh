#!/usr/bin/env bash
# One command on a fresh vast.ai box. No state survives the night, so this script
# IS the machine. Keep it in git; never configure a rented box by hand.
#
#   curl -sL <raw-url>/bootstrap/vast_setup.sh | bash
#   or:  bash bootstrap/vast_setup.sh
#
# Rent with VM MODE ON (cloud.vast.ai/create -> VM toggle). Unprivileged Docker
# containers do not expose hardware counters and `ncu` will fail with
# ERR_NVGPUCTRPERM. Verified 2026-08-12: VM mode has live RTX 5090 offers
# (~$0.40/hr) but ZERO H100/B200 offers -- do not assume ncu on datacenter tier.

set -euo pipefail
log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

log "identity"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv,noheader || { echo "no GPU visible"; exit 1; }

log "profiling permission (the thing that decides whether ncu works at all)"
if [ -r /proc/driver/nvidia/params ]; then
  grep -i RestrictProfiling /proc/driver/nvidia/params || echo "  (param absent)"
else
  echo "  /proc/driver/nvidia/params unreadable -- containerised, counters likely blocked"
fi

log "apt + python"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git build-essential python3-pip >/dev/null

log "torch + triton + fla (the baselines you are measuring against)"
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu128 || pip install -q torch
pip install -q triton flash-linear-attention einops numpy

log "clock lock -- without this every timing number is noise"
nvidia-smi -pm 1 >/dev/null 2>&1 || echo "  persistence mode unavailable"
MAXCLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
nvidia-smi --lock-gpu-clocks="${MAXCLK},${MAXCLK}" 2>/dev/null \
  && echo "  SM clocks locked at ${MAXCLK} MHz" \
  || echo "  CLOCK LOCK FAILED -- record this in the provenance line; timings will drift"

log "versions (paste this block into the ledger provenance line)"
python3 - <<'PY'
import torch, platform
print(f"  torch      {torch.__version__}")
print(f"  cuda       {torch.version.cuda}")
print(f"  device     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  sm         {p.major}{p.minor}   SMs {p.multi_processor_count}   "
          f"HBM {p.total_memory/1e9:.1f} GB")
try:
    import triton; print(f"  triton     {triton.__version__}")
except Exception: pass
try:
    import fla; print(f"  fla        {getattr(fla,'__version__','installed')}")
except Exception as e: print(f"  fla        MISSING ({e})")
print(f"  python     {platform.python_version()}")
PY

log "done. next: bash bootstrap/probe_ncu.sh"
