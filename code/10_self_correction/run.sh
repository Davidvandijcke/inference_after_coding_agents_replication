#!/usr/bin/env bash
# Table 9, Figures 9 and 10 and the in-text numbers on agents' own corrections. Reads records of modules 02 and 09.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
mkdir -p "$HERE/../../paper/figs"
"$PY" "$HERE/01_stance_table.py"
Rscript "$HERE/02_card_sol_interval.R"
"$PY" "$HERE/03_card_family.py"
"$PY" "$HERE/04_power_verdicts.py"
"$PY" "$HERE/05_numbers.py"
