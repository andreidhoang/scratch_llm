# GPU workflow — Vast.ai

The senior-engineer pattern: **code lives in GitHub, the pod is a disposable executor, big artifacts move out-of-band.** Nothing important ever lives only on a rented box.

```
laptop (source of truth)         Vast.ai pod (ephemeral)
  ├─ git push  ───────────────▶  git clone   (code + plan + context bundle)
  ├─ (provision)  ────────────▶  ~/.claude/ global config + lectures oracle
  └─ sync_checkpoints.sh ◀─────  checkpoints  (artifacts, never in git)
```

The pod reproduces your laptop's **exact workspace topology** so a pod agent loads the
*same plan and the same Claude Code context engineering* you have locally:

```
/root/cs336/
├── STRATEGY.md  DELTA.md  README.md   (vendored — the ../ refs in CLAUDE.md resolve)
├── lectures/                          (CS336 oracle, cloned from stanford-cs336)
└── scratch_llm/                       (repo; Claude auto-loads CLAUDE.md + docs/)
~/.claude/{CLAUDE,PRINCIPLES,RULES}.md (global SuperClaude config, installed by provision)
```

| Artifact | Channel | Why |
|---|---|---|
| Source code | GitHub (`git`) | versioned, reproducible, survives `destroy` |
| Plan + context | `deploy/context/` in git → provisioned | parity: pod agents know the plan |
| Oracle (`lectures/`) | `provision_lectures.sh` (public clone) | 452M, not bundled — cloned on demand |
| Datasets | rsync / cloud bucket | too big for git |
| Checkpoints | `sync_checkpoints.sh` / bucket | must survive preemption |
| Secrets | env vars only | never committed |

## One-time setup (laptop)

```bash
# 1. Wire up Vast (stores API key + uploads your SSH pubkey)
VAST_API_KEY=<from https://cloud.vast.ai/account> ./deploy/00_setup_vast.sh
```

`gh` is already authenticated, so the pod clones private repos using your existing
GitHub token automatically — no separate GitHub setup needed.

## Every session

```bash
# 2. (if you changed STRATEGY/DELTA/README or your global ~/.claude config)
#    refresh the vendored context bundle so the pod gets current files:
./deploy/sync_context.sh && git add deploy/context && git commit -m "context: refresh" && git push

# 3. Rent a GPU + provision (shows the offer & price before charging you)
GPU=H100 MAX_DPH=3.0 ./deploy/01_launch.sh
#    GPU=A100 MAX_DPH=1.5    -> cheaper
#    GPU=RTX4090 MAX_DPH=0.6 -> smoke tests
#    SKIP_LECTURES=1 ...     -> skip the oracle clone if you don't need adapter tests

# 4. SSH in (the launch script prints the exact command). Then:
#    cd /root/cs336/scratch_llm && source .venv/bin/activate
#    Launch Claude Code HERE so it auto-loads CLAUDE.md + the plan.

# 5. Pull checkpoints back periodically (resumable):
INSTANCE=<id> ./deploy/sync_checkpoints.sh

# 6. Done — STOP ALL BILLING (wipes disk):
vastai destroy instance <id>
```

## Billing — know the difference
- **Compute** ($/hr) is charged only while the instance is **running**.
- **Storage** is charged from `create` until `destroy` — *including while stopped*.
- `stop` ≠ `destroy`. Stop pauses compute but keeps paying for disk. **For community
  hosts, prefer `destroy` + re-provision from git** over `stop` (cheaper, and you don't
  risk losing the box). Re-provisioning is one `./deploy/01_launch.sh`.

## Files
| File | Runs on | Does |
|---|---|---|
| `00_setup_vast.sh` | laptop | one-time: API key + SSH key upload |
| `sync_context.sh` | laptop | refresh `context/` bundle from live laptop sources |
| `01_launch.sh` | laptop | search live offers → rent → wait for SSH → provision |
| `provision.sh` | pod | rebuild workspace topology + install global cfg + venv + smoke check |
| `provision_lectures.sh` | pod | clone the CS336 oracle (official stanford-cs336 repos) |
| `sync_checkpoints.sh` | laptop | rsync artifacts off the pod |
| `context/` | — | vendored plan + global config (see `context/README.md`) |
