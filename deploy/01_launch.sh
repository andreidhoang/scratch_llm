#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 01_launch.sh — rent a GPU, wait for SSH, provision the repo on it.
# Run on YOUR laptop, after 00_setup_vast.sh.
#
# Senior-engineer rules baked in:
#   - Offer IDs are NEVER hardcoded. We query the live marketplace each run and
#     pick the cheapest host matching the filter, because yesterday's host may
#     be gone today.
#   - We pass the GitHub token to the pod so it can clone the PRIVATE repo, but
#     the token never lands in the repo or in a file — only in the pod's env for
#     the duration of one SSH command.
#   - The pod is a disposable executor. Code lives in git; this script just
#     reconstitutes it on fresh metal.
#
# Tunables (env vars, with sensible defaults):
#   GPU=H100            # H100 | A100 | RTX4090 | RTX3090 ...
#   GPU_RAM=40          # min GPU RAM (GB)
#   MAX_DPH=3.0         # price ceiling ($/hr) — protects against bid spikes
#   DISK=80             # disk (GB)
#   REPO=andreidhoang/scratch_llm
#   GITHUB_TOKEN=ghp_... (or rely on `gh auth token`)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-H100}"
GPU_RAM="${GPU_RAM:-40}"
MAX_DPH="${MAX_DPH:-3.0}"
DISK="${DISK:-80}"
REPO="${REPO:-andreidhoang/scratch_llm}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel}"
PUBKEY="${SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"

# Reuse the laptop's gh login as the clone token unless one is passed explicitly.
GITHUB_TOKEN="${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}"
if [[ -z "$GITHUB_TOKEN" ]]; then
  echo "❌ No GitHub token. Run 'gh auth login' or set GITHUB_TOKEN." >&2
  exit 1
fi

FILTER="gpu_name=${GPU} gpu_ram>=${GPU_RAM} reliability>0.97 cuda_max_good>=12.0 \
inet_down>=200 disk_space>=${DISK} dph_total<${MAX_DPH} verified=true rentable=true"

echo "🔎 Searching offers: ${FILTER}"
OFFER=$(vastai search offers "$FILTER" -o 'dph_total' --raw | jq -r '.[0]')
if [[ -z "$OFFER" || "$OFFER" == "null" ]]; then
  echo "❌ No offers match. Loosen the filter (raise MAX_DPH or lower GPU_RAM)." >&2
  exit 1
fi
OFFER_ID=$(jq -r '.id'         <<<"$OFFER")
PRICE=$(jq    -r '.dph_total'  <<<"$OFFER")
GPU_SKU=$(jq  -r '.gpu_name'   <<<"$OFFER")
REL=$(jq      -r '.reliability'<<<"$OFFER")
echo "🏷  Picked offer ${OFFER_ID}: ${GPU_SKU}  \$${PRICE}/hr  reliability=${REL}"
read -rp "Rent this? Billing starts now. [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "Aborted."; exit 0; }

echo "🚀 Creating instance..."
CREATE=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK" --ssh --direct --raw)
INSTANCE_ID=$(jq -r '.new_contract' <<<"$CREATE")
echo "📦 Instance ${INSTANCE_ID} provisioning. Waiting for SSH..."

# Poll until the instance reports a connectable SSH endpoint.
for _ in $(seq 1 60); do
  URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null || true)
  if [[ -n "$URL" && "$URL" == ssh://* ]]; then
    HOSTPORT="${URL#ssh://}"            # root@1.2.3.4:12345
    PORT="${HOSTPORT##*:}"
    HOST="${HOSTPORT%:*}"
    if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 \
           -p "$PORT" "$HOST" true 2>/dev/null; then
      break
    fi
  fi
  sleep 10
done
[[ -n "${HOST:-}" ]] || { echo "❌ SSH never came up. Check 'vastai show instances'." >&2; exit 1; }
echo "🔌 SSH up: ssh -p ${PORT} ${HOST}"

echo "🛠  Provisioning repo on the pod..."
scp -P "$PORT" -o StrictHostKeyChecking=accept-new deploy/provision.sh "$HOST:/root/provision.sh"
ssh -p "$PORT" "$HOST" \
  "REPO='$REPO' GITHUB_TOKEN='$GITHUB_TOKEN' bash /root/provision.sh"

cat <<EOF

✅ READY.
   Instance  : ${INSTANCE_ID}   ${GPU_SKU}  \$${PRICE}/hr
   SSH       : ssh -p ${PORT} ${HOST}
   Workspace : /root/cs336  (lectures oracle + scratch_llm; the plan travels IN the repo: PLAN.md)
   Code      : /root/cs336/scratch_llm  (pip-installed [gpu,dev]; same plan/context as laptop)
   ⚠ ncu    : this is a plain Vast DOCKER pod — profiling counters are BLOCKED (ERR_NVGPUCTRPERM).
              For ncu work rent a vms_enabled KVM instead (PLAN.md § Hardware law).
   Agents    : launch Claude Code from /root/cs336/scratch_llm — auto-loads CLAUDE.md + the plan

   Pull checkpoints back to your laptop any time:
     INSTANCE=${INSTANCE_ID} ./deploy/sync_checkpoints.sh

   When done (STOPS ALL BILLING — disk wiped):
     vastai destroy instance ${INSTANCE_ID}
EOF
