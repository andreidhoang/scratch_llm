#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 00_setup_vast.sh — one-time Vast.ai account wiring (run on YOUR laptop).
#
# Does two things, both idempotent:
#   1. Stores your Vast.ai API key so the `vastai` CLI can authenticate.
#   2. Uploads your SSH *public* key to Vast so every pod you launch trusts it.
#
# WHY upload the key BEFORE launching: on Vast VMs the authorized key is baked
# in at boot and "cannot be edited on a running VM" — wrong key == you can never
# SSH in. Pods are more forgiving, but doing it once up front removes the class
# of failure entirely.
#
# Usage:
#   VAST_API_KEY=xxxxx ./deploy/00_setup_vast.sh
#   (or omit the env var and the script will prompt you, without echoing it)
# ---------------------------------------------------------------------------
set -euo pipefail

PUBKEY="${SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"

# --- 1. API key -----------------------------------------------------------
if [[ -z "${VAST_API_KEY:-}" ]]; then
  read -rsp "Paste your Vast.ai API key (https://cloud.vast.ai/account): " VAST_API_KEY
  echo
fi
vastai set api-key "$VAST_API_KEY"
echo "✅ Vast API key stored (~/.config/vastai/)."

# --- 2. SSH key -----------------------------------------------------------
if [[ ! -f "$PUBKEY" ]]; then
  echo "❌ No public key at $PUBKEY. Generate one: ssh-keygen -t ed25519" >&2
  exit 1
fi

# Upload only if this exact key isn't already registered (keeps it idempotent).
if vastai show ssh-keys 2>/dev/null | grep -qF "$(cut -d' ' -f2 < "$PUBKEY")"; then
  echo "✅ SSH key already registered with Vast."
else
  vastai create ssh-key "$(cat "$PUBKEY")"
  echo "✅ Uploaded $PUBKEY to Vast."
fi

echo
echo "Done. Next: ./deploy/01_launch.sh  (rents a GPU and provisions it)."
