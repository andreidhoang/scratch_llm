# Claude Code auto-memory — durable snapshot (survives pod destroy)

**Why this exists.** Claude Code's per-project auto-memory lives at
`~/.claude/projects/<slug>/memory/` — **outside** this git repo, on the container filesystem. A
Vast.ai **recycle/destroy wipes it** (and renting a *new* pod is a fresh machine). GitHub is the
only guaranteed-durable store, so the canonical copy of every memory lives **here, in the repo**,
and is restored to the live location on a fresh pod by `scripts/bootstrap-pod.sh`.

**Source of truth = this folder.** When memory changes during a session, re-sync it here:
```bash
cp ~/.claude/projects/*/memory/*.md .claude/memory-snapshot/   # then commit
```
(The bootstrap script copies the other direction on a new pod: snapshot → live.)

**Files**
- `MEMORY.md` — the index Claude loads each session (one line per memory).
- `*.md` (others) — one fact each (user prefs, feedback, project state) with frontmatter.

**Restore manually** (if not using the script): copy these into your session's memory dir —
find it with `ls ~/.claude/projects/` — or just tell a fresh Claude:
> "Read `.claude/memory-snapshot/` and re-save each as a memory."

See `docs/VASTAI_BOOTSTRAP.md` for the full fresh-pod runbook.
