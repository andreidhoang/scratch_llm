#!/usr/bin/env bash
# PreToolUse(Bash git commit) — ENFORCEMENT hook (Lever 4).
# Blocks a commit (exit 2) if CI is red. A red commit must be impossible, not merely
# discouraged — that is the whole point of putting this in structure, not in CLAUDE.md prose.
# Graceful on an empty clean room: no git → skip; no tests → pass (pytest exit 5).
set -uo pipefail

# Scope. This script has TWO callers and must recognise both:
#
#   Claude Code  --PreToolUse-->  stdin = {"tool_input":{"command":"..."}}
#   git          --pre-commit-->  stdin = EMPTY          <-- .git/hooks/pre-commit symlink
#
# It previously matched the literal string "git com<mit>" anywhere in stdin, which was wrong in
# BOTH directions: git supplies no stdin at all, so every terminal commit fell through to exit 0
# (the symlink had never blocked anything — that is how a red instrument commit shipped), while
# any Bash call whose TEXT merely mentioned committing — a heredoc writing documentation, say —
# ran the full gate and was refused. Both bugs are one bug: inferring intent from a substring.
# Fixed by reading the caller from structure instead: git sets GIT_INDEX_FILE for its hooks and
# invokes them by a path under .git/hooks/; the harness supplies parseable JSON.
# PHASE SPLIT (2026-08-26). The gate runs in two speeds, because a slow gate is a bypassed gate:
#
#   commit -> ruff + pyright only          ~10s   fast enough that nobody reaches for Ctrl-C
#   push   -> the above + the full suite   ~4-5m  runs once per push, and push is what the
#                                                 public remote (and therefore "shipped") means
#
# Why this matters beyond ergonomics: while `git commit` holds .git/index.lock, an interrupted
# commit leaves that lock behind and every later git call dies with "Unable to create index.lock".
# A 4-minute blocking commit manufactures exactly that. Two stale locks were cleared on
# 2026-08-26 (index.lock 19/08 21:24 — right at the 21:00 wall; objects/maintenance.lock 21/08).
# Recovery, if one ever reappears: confirm no git process is live (`ps ax | grep git`), then
# `rm -f .git/index.lock`.
#
# Caller detection is by STRUCTURE, never by substring. $0 is authoritative when git invokes us
# through .git/hooks/<name>; note pre-push receives ref lines on stdin, so "empty stdin" alone
# cannot distinguish the callers. The harness path parses JSON and ignores heredoc bodies, so
# merely *writing about* a commit is never mistaken for making one.
payload="$(cat 2>/dev/null || true)"
phase=""

case "$(basename -- "$0")" in
  pre-commit) phase=commit ;;
  pre-push)   phase=push ;;
  *)
    if [ -n "$payload" ]; then
      phase="$(printf '%s' "$payload" | python3 -c '
import json, re, sys
try:
    cmd = (json.load(sys.stdin).get("tool_input") or {}).get("command") or ""
except Exception:
    print(""); raise SystemExit
cmd = re.sub(r"<<-?\s*[\"\x27]?(\w+)[\"\x27]?.*?^\s*\1\s*$", " ", cmd, flags=re.S | re.M)
sep = r"(?:^|[;&|]|\n)\s*git\s+"
print("push" if re.search(sep + r"push\b", cmd) else
      "commit" if re.search(sep + r"commit\b", cmd) else "")
' 2>/dev/null || echo "")"
    elif [ -n "${GIT_INDEX_FILE:-}" ]; then
      phase=commit                     # git hook env present but an unexpected hook name
    fi
    ;;
esac

[ -n "$phase" ] || exit 0              # not a commit and not a push -> allow untouched

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

# Scope and marker set MIRROR .github/workflows/ci.yml exactly. They had drifted: the hook checked
# `src tests` while CI checks `src tests bench/kernels`, and the hook ran the whole `not gpu` set
# while CI deselects `slow`. A gate that is not identical to CI can pass locally and fail on the
# remote, which is the same class of gap that let matplotlib through. Keep these two in lockstep.
if command -v ruff >/dev/null 2>&1; then
  ruff check src tests bench/kernels       >/dev/null 2>&1 || fail "ruff check failed"
  ruff format --check src tests bench/kernels >/dev/null 2>&1 || fail "ruff format would change files (run: ruff format)"
fi

command -v pyright >/dev/null 2>&1 && { pyright >/dev/null 2>&1 || fail "pyright reported type errors"; }

if [ "$phase" = "push" ] && command -v pytest >/dev/null 2>&1; then
  # PUSH ONLY. -x stops at the first failure so a red push fails in seconds, not minutes.
  pytest -m "not gpu and not slow" -q -x >/dev/null 2>&1
  rc=$?
  # 0 = passed, 5 = no tests collected (fine in an empty clean room)
  [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ] || fail "pytest failed (exit $rc) — run it yourself to see which"
elif [ "$phase" = "commit" ]; then
  echo "green-ci-gate: lint+types OK (suite deferred to pre-push)." >&2
fi

# FOP-1 doc-discipline — WARN-ONLY (never blocks, never changes exit status).
# If the staged set adds a new .md but stages no code, nudge: ship a node, don't pile docs.
staged="$(git diff --cached --name-only --diff-filter=A 2>/dev/null || true)"
if echo "$staged" | grep -Eq '\.md$' && ! echo "$staged" | grep -Eq '\.(py|cu|triton)$'; then
  echo "⚠ FOP-1: staged a new .md with no staged code — close a node, don't write the next doc (CLAUDE.md FOP)." >&2
fi

exit 0
