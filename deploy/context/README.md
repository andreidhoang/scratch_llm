# Context bundle — project plan & roadmap parity

This directory makes the repo **self-describing for this project**: a freshly-pulled
clone (on a GPU pod or any machine) carries the same project plan, context
engineering, and roadmap — without depending on files that live in the parent dir
outside the repo.

```
context/
└── workspace/        # the parent cs336/ docs the repo's CLAUDE.md links via ../
    ├── STRATEGY.md   #   the canonical ship-order / roadmap (CLAUDE.md: "wins over any other doc")
    ├── DELTA.md      #   the capstone spec
    └── README.md     #   the workspace map
```

**Scope:** this project's roadmap/context only. The repo's own `CLAUDE.md`, `docs/`
(`IMPLEMENTATION_PLAN.md`, `STATUS.md`, `CONTEXT_ENGINEERING.md`, guides, ADRs), and
`.claude/` (commands `/master` `/next`, hooks) already travel with `git` — this bundle
just adds the three parent-dir docs that `../`-links would otherwise miss. Your global
`~/.claude` setup (personal config, skills, plugins) is **per-machine** and is not
carried here.

## How it's used
`deploy/provision.sh` (on the pod) lays `workspace/*` into `/root/cs336/`. Since the
repo clones to `/root/cs336/scratch_llm`, the repo's `../STRATEGY.md` etc. resolve
exactly as on your laptop — **no link rewriting**.

## Keeping it fresh (avoid drift)
This is a **vendored snapshot** of the parent-dir docs. When you edit them on your
laptop, refresh the bundle before launching a pod:

```bash
./deploy/sync_context.sh          # copies ../STRATEGY.md etc. -> here, shows the diff
git add deploy/context && git commit -m "context: refresh roadmap docs" && git push
```
