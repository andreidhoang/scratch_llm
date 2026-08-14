#!/usr/bin/env bash
# Settles P-000: "will a rented instance give working ncu counters?"
# Open since 2026-08-12, predicted YES, never actually tested. Binary, ~90 seconds.
#
# This is the day's cannot-fail opener. It cannot fail because both outcomes are
# useful: counters work -> the whole profiling half of the plan is unlocked;
# counters blocked -> nsys + manual timing is the honest fallback and every
# ledger row from this box carries that in its provenance line instead of
# silently degrading.

set -uo pipefail
log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

log "is ncu even installed?"
if ! command -v ncu >/dev/null 2>&1; then
  for c in /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu; do
    [ -x "$c" ] && export PATH="$(dirname "$c"):$PATH" && break
  done
fi
command -v ncu >/dev/null 2>&1 \
  && ncu --version | head -2 \
  || { echo "  ncu NOT FOUND -> apt-get install -y nsight-compute, or fall back to nsys"; }

log "trivial kernel (a memcpy -- known bandwidth, so the counter has a right answer)"
cat > /tmp/probe.py <<'PY'
import torch
n = 1 << 26                       # 64 Mi elements = 256 MB fp32 each side
a = torch.empty(n, device="cuda", dtype=torch.float32).normal_()
b = torch.empty_like(a)
for _ in range(3):
    b.copy_(a)
torch.cuda.synchronize()
import time
t = time.perf_counter()
for _ in range(20):
    b.copy_(a)
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / 20
gb = 2 * a.numel() * a.element_size() / 1e9
print(f"  copy {gb:.2f} GB in {dt*1e3:.3f} ms  ->  {gb/dt:.1f} GB/s achieved")
PY
python3 /tmp/probe.py

log "the actual test: can ncu read hardware counters?"
OUT=$(ncu --set basic --target-processes all python3 /tmp/probe.py 2>&1)
echo "$OUT" | tail -25

echo
if echo "$OUT" | grep -q "ERR_NVGPUCTRPERM"; then
  printf '\033[1;31m  P-000 = NO.\033[0m Counters blocked (ERR_NVGPUCTRPERM).\n'
  printf '  This is a HOST setting, not fixable from inside. Options, in order:\n'
  printf '    1. re-rent with VM mode ON (cloud.vast.ai/create -> VM toggle)\n'
  printf '    2. fall back to nsys + manual timing, and SAY SO in every provenance line\n'
elif echo "$OUT" | grep -qiE "dram__|gpu__time|Duration|Memory Throughput"; then
  printf '\033[1;32m  P-000 = YES.\033[0m Hardware counters are live.\n'
  printf '  Compare the ncu DRAM throughput against the wall-clock GB/s above.\n'
  printf '  They should agree within a few percent. If they do not, ONE of your two\n'
  printf '  instruments is lying and you need to know which before trusting either.\n'
else
  printf '\033[1;33m  P-000 = INCONCLUSIVE.\033[0m Paste the full output back.\n'
fi
