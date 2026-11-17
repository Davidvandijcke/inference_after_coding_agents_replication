#!/usr/bin/env bash
# Table 1 Panel C, the real-data bar of Figure 3(a) (drawn by module 02) and the placebo design's in-text numbers.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
Rscript "$HERE/01_build_base_files.R"
"$PY" "$HERE/02_make_placebo_datasets.py"
"$PY" "$HERE/03_replay_counts.py"
Rscript "$HERE/04_replay_correction.R"
"$PY" "$HERE/05_identity_check.py"
"$PY" "$HERE/06_panel_c.py"
