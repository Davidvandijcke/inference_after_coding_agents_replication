#!/usr/bin/env bash
# Table 1 Panel B, Figure 3 and the open-ended design's in-text numbers. Run modules 01 and 03 first.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
"$PY" "$HERE/01_replay_counts.py"
Rscript "$HERE/02_replay_intervals.R"
"$PY" "$HERE/03_panel_b_figure.py"
