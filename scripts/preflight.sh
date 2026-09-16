#!/usr/bin/env bash
# Run the repository's deterministic checks before a commit or PR update.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
T4_CHECK_PYTHON="${T4_PYTHON:-python3}"

"$T4_CHECK_PYTHON" -m ruff check baselines scoring faithfulness qfbench2_track_analysis
"$T4_CHECK_PYTHON" -m ruff format --check scoring faithfulness qfbench2_track_analysis
"$T4_CHECK_PYTHON" -m pytest baselines scoring faithfulness tests -q
"$T4_CHECK_PYTHON" .github/validate_units.py analysis
