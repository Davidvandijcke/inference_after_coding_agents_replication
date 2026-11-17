#!/usr/bin/env bash
# Table 1 Panel D and the power design's in-text numbers.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
"$PY" "$HERE/01_prespecified.py"
"$PY" "$HERE/02_replay_counts.py"
Rscript "$HERE/03_replay_bounds.R"
"$PY" "$HERE/04_panel_d.py"
