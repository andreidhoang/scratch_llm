#!/usr/bin/env bash
# PreToolUse(Bash git commit) — ENFORCEMENT hook (Lever 4).
# Blocks a commit (exit 2) if CI is red. A red commit must be impossible, not merely
# discouraged — that is the whole point of putting this in structure, not in CLAUDE.md prose.
# Graceful on an empty clean room: no git → skip; no tests → pass (pytest exit 5).
set -uo pipefail

# Scope (the documented intent): gate ONLY an actual `git commit`. PreToolUse passes the proposed
# tool call as JSON on stdin; the settings.json `if: Bash(git commit*)` matcher is not honored by
# this harness (the gate was firing on every Bash), so enforce the scope here. Any non-commit Bash
# is allowed straight through — only a red *commit* is blocked.
payload="$(cat 2>/dev/null || true)"
case "$payload" in
  *"git commit"*) : ;;   # a commit → run the green gate below
  *) exit 0 ;;           # any other Bash → allow untouched
esac

root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$root" ] && exit 0          # not a git repo yet → nothing to gate
cd "$root"

# Prefer the project venv's tools — bare ruff/pyright/pytest otherwise resolve to a system Python
# without the deps (torch/regex), which would spuriously fail a valid commit. The venv is the
# source of truth (it is what `uv pip install -e ".[dev]"` populates).
[ -x "$root/.venv/bin/python" ] && PATH="$root/.venv/bin:$PATH"

fail() {
  echo "green-ci-gate: BLOCKED — $1." >&2
  echo "Fix it before committing (clean-room green-CI rule; see CLAUDE.md)." >&2
  exit 2                          # exit 2 on PreToolUse blocks the tool call
}

if command -v ruff >/dev/null 2>&1; then
  ruff check src tests       >/dev/null 2>&1 || fail "ruff check failed"
  ruff format --check src tests >/dev/null 2>&1 || fail "ruff format would change files (run: ruff format)"
fi

command -v pyright >/dev/null 2>&1 && { pyright >/dev/null 2>&1 || fail "pyright reported type errors"; }

if command -v pytest >/dev/null 2>&1; then
  pytest -m "not gpu" -q >/dev/null 2>&1
  rc=$?
  # 0 = passed, 5 = no tests collected (fine in an empty clean room)
  [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ] || fail "pytest failed (exit $rc)"
fi

# FOP-1 doc-discipline — WARN-ONLY (never blocks, never changes exit status).
# If the staged set adds a new .md but stages no code, nudge: ship a node, don't pile docs.
staged="$(git diff --cached --name-only --diff-filter=A 2>/dev/null || true)"
if echo "$staged" | grep -Eq '\.md$' && ! echo "$staged" | grep -Eq '\.(py|cu|triton)$'; then
  echo "⚠ FOP-1: staged a new .md with no staged code — close a node, don't write the next doc (CLAUDE.md FOP)." >&2
fi

exit 0
