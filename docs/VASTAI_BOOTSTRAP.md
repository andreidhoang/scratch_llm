# Vast.ai bootstrap — reconstitute the whole thing on a fresh pod

> **What survives a pod destroy, what doesn't** (see the Vast agent guide in `/etc/vast_agents/`):
> a recycle/destroy **wipes the container filesystem**; only a mounted host **volume** persists, and
> a *newly rented* pod is a different machine. **Treat GitHub as the only durable store.** Everything
> needed to get back to a working state is committed here — this runbook is the reassembly order.

## What's durable vs what you rebuild

| Thing | Durable? | How it comes back |
|---|---|---|
| Source, tests, `bench/RESULTS.md`, all `docs/` | ✅ in git | `git clone` |
| `.claude/` harness (hooks, agents, commands, settings) | ✅ in git (21 files) | auto-loads when you open Claude Code here |
| **Claude Code auto-memory** (user prefs/feedback) | ⚠️ lives in `~/.claude/`, **NOT in git** | **no longer mirrored** — the snapshot dir was a stale duplicate, deleted 31/08; re-create it only if a pod actually needs it |
| `.venv` (torch/triton/deps) | ❌ container FS | `uv` from `uv.lock` (step 2) |
| git pre-commit hook (`.git/hooks/`) | ❌ never cloned | re-symlink (step 4) |
| GitHub auth, git identity | ❌ secrets, per-machine | `gh auth login` (step 3) |
| Claude Code binary + user settings (`/model`, `/effort`) | ❌ per-machine | install (step 1) + re-set |

## The measured stack (what this repo was built and profiled on)

- GPU: **NVIDIA RTX PRO 4000 Blackwell, sm_120, 24 GB, 0.55 TB/s** (any sm_120 / RTX-50xx is equivalent).
- **torch `2.12.1+cu130`, triton `3.7.1`** — sm_120 needs a **CUDA ≥ 12.8** wheel; the default
  `uv pip install torch` may pull a build with no sm_120 kernels (`no kernel image` at first GPU op).
  Pin the cu130 index (step 2). *On a different GPU, drop the `--index-url` and let uv pick.*
- Python 3.12; `uv` for envs.

---

## Fresh pod, in order

### 0. Rent + open a shell
Rent an sm_120 (or your target-tier) instance on Vast. If `/workspace` is a **volume you re-attached**,
the repo may already be there — `cd /workspace/scratch_llm && git pull`. Otherwise clone fresh.

### 1. Install Claude Code
```bash
curl -fsSL https://claude.ai/install.sh | bash     # native installer → ~/.local/bin/claude
#   (or, with node ≥ 18:  npm install -g @anthropic-ai/claude-code)
claude --version
```
On this base image node is via nvm and only on PATH in a **login shell** — if `npm` isn't found:
`. /opt/nvm/nvm.sh` first (see the Vast guide §2).

### 2. Clone + one-command bootstrap
```bash
cd /workspace
git clone https://github.com/andreidhoang/scratch_llm.git && cd scratch_llm
bash scripts/bootstrap-pod.sh      # env + torch cu130 + hook re-link + memory restore + verify
```
The script prints the current curriculum node at the end. If it reports torch has no CUDA or the
gate is red, fix before working (usually the torch wheel — re-run step 2's torch line for your GPU).

### 3. Authenticate git + GitHub (needs your input — not scriptable)
```bash
gh auth login                                              # GitHub.com → HTTPS → browser/paste token
git config credential.helper '!gh auth git-credential'    # so `git push` just works
git config user.name  "Huy Hoang Dang"
git config user.email "danghuy19990804@gmail.com"
```

### 4. (only if you skipped the script) manual equivalents
```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130
ln -sf ../../.claude/hooks/green-ci-gate.sh .git/hooks/pre-commit         # green-CI on every commit
mkdir -p ~/.claude/projects/-workspace/memory
# (memory-snapshot mirror removed 31/08 — nothing to restore)
```

### 5. Open Claude Code and orient
```bash
cd /workspace/scratch_llm && claude
#  · CLAUDE.md + the SessionStart hook auto-load → you'll see the current node injected.
#  · re-set session prefs if you want: /model  → Fable 5 ,  /effort → max
#  · then run  /standup  (or /next) — it reads PLAN.md + KERNEL_MASTERY_SPEC §12.5 + bench/RESULTS.md.
```

## Where "the next task" lives (so a fresh Claude self-orients)

Nothing depends on memory — the state is on disk and pushed:
`PLAN.md` (§ TODAY = the current node) → `bench/RESULTS.md` (measured ledger) →
`MASTERY_LEDGER.md` (the review draft). Full mechanism: **`docs/CONTEXT_ENGINEERING.md`**.

## Memory on a pod (snapshot mirror removed 31/08)

`~/.claude/projects/*/memory/` is per-machine and is **not** carried by this repo — the in-repo
mirror was a stale duplicate and was deleted. Nothing in the workflow depends on it: the state a
fresh session needs is on disk and pushed (`PLAN.md` → `bench/RESULTS.md` → `MASTERY_LEDGER.md`).
