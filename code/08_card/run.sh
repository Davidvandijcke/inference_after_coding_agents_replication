#!/usr/bin/env bash
# Figures 5 to 8, Tables 6 and 7 and the Card-Krueger application's in-text numbers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"
export PYTHONDONTWRITEBYTECODE=1
Rscript "$HERE/01_original.R"
Rscript "$HERE/02_bank.R"
Rscript "$HERE/03_correction.R"
"$PY" "$HERE/04_threshold.py"
"$PY" "$HERE/05_selection_figure.py"
Rscript "$HERE/06_choice_pvalues.R"
"$PY" "$HERE/07_choice_figure.py"
"$PY" "$HERE/08_replay_figure.py"
"$PY" "$HERE/09_activity.py"
"$PY" "$HERE/10_variables_table.py"
"$PY" "$HERE/11_numbers.py"
