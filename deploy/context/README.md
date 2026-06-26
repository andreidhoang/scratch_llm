# Context bundle — parity across machines

This directory makes the repo **self-describing**: a freshly-provisioned GPU pod
gets the *same* plan and the *same* Claude Code context engineering as your laptop,
without depending on files that live outside the repo.

```
context/
├── workspace/        # the parent cs336/ docs the repo's CLAUDE.md links via ../
│   ├── STRATEGY.md   #   the canonical ship-order (CLAUDE.md: "wins over any other doc")
│   ├── DELTA.md      #   the capstone spec
│   └── README.md     #   the workspace map
└── claude_global/    # the global ~/.claude SuperClaude config (loads every turn)
    ├── CLAUDE.md     #   entry point (@-imports the two below)
    ├── PRINCIPLES.md #   engineering principles
    └── RULES.md      #   operational rules
```

## How it's used
`deploy/provision.sh` (on the pod) lays these out so paths resolve identically:
- `workspace/*` → `/root/cs336/` (the repo clones to `/root/cs336/scratch_llm`, so
  `../STRATEGY.md` etc. resolve exactly as on your laptop — **no link rewriting**).
- `claude_global/*` → `~/.claude/` (so PRINCIPLES/RULES + the learning methodology
  load on every turn, same as your laptop).

## Keeping it fresh (avoid drift)
This is a **vendored snapshot**. When you edit the live sources on your laptop,
refresh the bundle before launching a pod:

```bash
./deploy/sync_context.sh          # copies live sources -> here, shows the diff
git add deploy/context && git commit -m "context: refresh bundle" && git push
```
