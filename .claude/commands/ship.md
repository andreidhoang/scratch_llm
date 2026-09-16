---
description: Review the current diff against the active v5 contract and required checks; report readiness and prepare a commit message.
argument-hint: "(optional) short note on what this change is"
---
Ship review for the current change. $ARGUMENTS

1. Read repository AGENTS.md/CLAUDE.md and the active ladders v5 contract. Inspect git diff and
   git diff --staged.
2. Route to ship-reviewer for correctness, evidence and requested scope.
3. Run the checks appropriate to the change, using CLAUDE.md's actual CI commands and the active
   rung's runtime gates. Distinguish executed checks, skips and unmet requirements. Existing
   hooks have missing-tool and marker-scope limitations; their presence is not a test result.
4. Report ACCEPT or the concrete fix list, and prepare a commit message. Commit/push only when
   authorized by the current task; this review command and an old execution-mode label do not
   automatically grant publication authority. Respect ownership when implementing any fixes.
