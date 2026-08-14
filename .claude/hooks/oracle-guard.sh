#!/usr/bin/env bash
# PreToolUse(Edit|Write) — the mastery boundary, enforced by tooling rather than willpower.
# Blocks agent writes to the short list in CLAUDE.md. Everything else passes.
# Fail-open on parse errors (exit 0) so a malformed payload never blocks legitimate work.
set -uo pipefail
payload="$(cat 2>/dev/null || true)"
path="$(printf '%s' "$payload" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: print("")' 2>/dev/null || true)"

case "$path" in
  */mastery/reference.py)
    echo "[MASTERY BOUNDARY] mastery/reference.py is THE ORACLE — the human writes it from blank." >&2
    echo "Every number in this repo is a deviation from it in float64. Five lines; the equation is" >&2
    echo "in its docstring. Explain the mechanism, do not write the body." >&2
    exit 2 ;;
  */mastery/harness*.py | */bench/harness.py)
    echo "[MASTERY BOUNDARY] the measurement harness is the human's. You do not own a reading" >&2
    echo "from an instrument you did not build. Plumbing around it is yours; the timing loop is not." >&2
    exit 2 ;;
  */mastery/losses.py | */mastery/grpo*.py | */mastery/verifier*.py | *_kernel.py | *.cu | *.cuh)
    echo "[MASTERY BOUNDARY] RL loss math, verifier logic, and kernel bodies are the human's." >&2
    echo "(CLAUDE.md 'What agents must NOT write here'.) Offer a failing test, a mechanism, or an" >&2
    echo "L2 menu of variants with the ranking hidden." >&2
    exit 2 ;;
  *) exit 0 ;;
esac
