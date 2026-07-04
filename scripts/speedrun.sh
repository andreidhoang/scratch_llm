#!/usr/bin/env bash
# Close-the-loop speedrun spine (ADR-0018 / docs/FRONTIER_2026_ABLATIONS.md §5).
# The real logic lives in src/scratch_llm/speedrun.py so it is importable + tested; this is the CLI.
#
#   scripts/speedrun.sh --nano                       # Phase-0 pre-flight (seconds, CPU)
#   scripts/speedrun.sh --depth 20 --device cuda ... # the d20 headline (8xH100 rental)
set -euo pipefail
cd "$(dirname "$0")/.."
# Prefer the project venv's python; fall back to whatever python is on PATH.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
exec python -m scratch_llm.speedrun "$@"
