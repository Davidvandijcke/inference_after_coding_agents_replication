#!/usr/bin/env bash
# Table 1 Panel A, the fixed-rule bars of Figure 3(a) (drawn by module 02) and the fixed design's in-text numbers.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
Rscript "$HERE/01_replay_estimates.R"
"$PY" "$HERE/02_panel_a.py"
