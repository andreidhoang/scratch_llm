#!/usr/bin/env bash
# Meat-boundary backstop (PreToolUse on Edit|Write).
# Claude/agents must NOT write kernel IMPLEMENTATIONS — the human reconstructs them from blank.
# Blocks Edit/Write to kernel-rep files; the bench harness, tests, oracle, and __init__ are NOT blocked
# (agents may help with those). Add new kernel files to the case patterns as you create them.
# Fail-open: a parse error never blocks legitimate work (exit 0).
set -uo pipefail

payload="$(cat 2>/dev/null || true)"
path="$(printf '%s' "$payload" | python3 -c 'import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get("tool_input", {}).get("file_path", ""))
except Exception:
    print("")' 2>/dev/null || true)"

case "$path" in
  */src/scratch_llm/kernels/matmul.py | */src/scratch_llm/kernels/*_triton.py | *_kernel.py)
    echo "[MEAT BOUNDARY] $path is a kernel implementation — you reconstruct it from blank." >&2
    echo "Claude won't write it. Use /tutor for the mechanism, then write it in your own editor." >&2
    exit 2
    ;;
  *)
    exit 0
    ;;
esac
