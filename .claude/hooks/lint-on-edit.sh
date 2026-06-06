#!/usr/bin/env bash
# PostToolUse(Edit|Write) — CONVENIENCE hook (Lever 4). Lints+formats the just-edited
# Python file so style never costs the model (or you) a thought. Never blocks: exit 0 always.
# Reads the tool-call JSON on stdin; extracts file_path with python3 (no jq dependency).
set -uo pipefail

fp="$(python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception: print("")' 2>/dev/null || true)"

case "$fp" in
  *.py)
    command -v ruff >/dev/null 2>&1 || { echo "lint-on-edit: ruff not installed; skipped" >&2; exit 0; }
    ruff check --fix "$fp"  >&2 2>&1 || true   # autofix what is safe; surface the rest, don't block
    ruff format "$fp"       >&2 2>&1 || true
    ;;
esac
exit 0
