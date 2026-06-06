---
description: Pre-commit gate — review the diff via ship-reviewer + green-CI, then hand me the commit command
argument-hint: "(optional) short note on what this change is"
---
Ship review for the current change. $ARGUMENTS

1. Show me the diff — run `git diff` and `git diff --staged` (read-only).
2. Route the diff to the **ship-reviewer** subagent: correctness + design/scope against the
   layer target (the `CLAUDE.md` layer→file map and the relevant `docs/assignment_guides/*`).
3. If **ACCEPT**: run the green-CI checks (`ruff check src tests`, `ruff format --check src tests`,
   `pyright`, `pytest -m "not gpu"`) and report results. Then propose the exact commit message in the
   form `<area>: <imperative>` and present the `git commit` command for ME to run — do NOT run
   `git commit` yourself (commits require my explicit action). The green-ci-gate hook re-checks on commit.
4. If **REJECT**: list the ordered fixes and stop. Do not proceed to commit.
