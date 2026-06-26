# GPU workflow — Vast.ai

The senior-engineer pattern: **code lives in GitHub, the pod is a disposable executor, big artifacts move out-of-band.** Nothing important ever lives only on a rented box.

```
laptop (source of truth)         Vast.ai pod (ephemeral)
  ├─ git push  ───────────────▶  git clone   (code)
  └─ sync_checkpoints.sh ◀─────  checkpoints  (artifacts, never in git)
```

| Artifact | Channel | Why |
|---|---|---|
| Source code | GitHub (`git`) | versioned, reproducible, survives `destroy` |
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
# 2. Rent a GPU + provision (shows the offer & price before charging you)
GPU=H100 MAX_DPH=3.0 ./deploy/01_launch.sh
#    GPU=A100 MAX_DPH=1.5   -> cheaper
#    GPU=RTX4090 MAX_DPH=0.6 -> smoke tests

# 3. SSH in (the launch script prints the exact command), train.
#    Code is at /root/scratch_llm, already `uv pip install -e ".[gpu,dev]"`.

# 4. Pull checkpoints back periodically (resumable):
INSTANCE=<id> ./deploy/sync_checkpoints.sh

# 5. Done — STOP ALL BILLING (wipes disk):
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
| `01_launch.sh` | laptop | search live offers → rent → wait for SSH → provision |
| `provision.sh` | pod | clone repo + `uv` venv + install + GPU smoke check |
| `sync_checkpoints.sh` | laptop | rsync artifacts off the pod |
