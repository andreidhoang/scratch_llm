#!/usr/bin/env bash
# Meat-boundary backstop (PreToolUse on Edit|Write) — governed by the execution-mode switch (ADR-0013).
#   .claude/execution-mode == "learn"    → block agent writes to kernel-rep files (the human
#                                          reconstructs them from blank; the historical contract).
#   .claude/execution-mode == "delegate" → allow (agents implement kernels end-to-end; correctness
#                                          is gated by oracle tests + adversarial review, not by
#                                          who typed the code).
# The bench harness, tests, oracle, and __init__ are never blocked in either mode.
# Fail-open: a parse error never blocks legitimate work (exit 0).
# Reversible in one line: write "learn" to .claude/execution-mode to re-arm the block.
set -uo pipefail

payload="$(cat 2>/dev/null || true)"
path="$(printf '%s' "$payload" | python3 -c 'import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get("tool_input", {}).get("file_path", ""))
except Exception:
    print("")' 2>/dev/null || true)"

# Resolve the mode: default to "learn" (the strict contract) if the switch file is missing.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
mode_file="${CLAUDE_PROJECT_DIR:-${script_dir}/../..}/.claude/execution-mode"
[ -f "$mode_file" ] || mode_file="${script_dir}/../execution-mode"
mode="$(tr -d '[:space:]' < "$mode_file" 2>/dev/null || echo learn)"

if [ "$mode" = "delegate" ]; then
  exit 0   # ADR-0013: full delegation — kernel writes allowed; oracle tests + review gate instead
fi

case "$path" in
  */src/scratch_llm/kernels/matmul.py | */src/scratch_llm/kernels/*_triton.py | *_kernel.py)
    echo "[MEAT BOUNDARY] $path is a kernel implementation — you reconstruct it from blank." >&2
    echo "Claude won't write it. Use /tutor for the mechanism, then write it in your own editor." >&2
    echo "(To lift this for full agent delegation, set .claude/execution-mode to 'delegate' — ADR-0013.)" >&2
    exit 2
    ;;
  *)
    exit 0
    ;;
esac
