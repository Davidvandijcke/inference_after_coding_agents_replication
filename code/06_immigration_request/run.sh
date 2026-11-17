#!/usr/bin/env bash
# Figure 4, Table 3 and the in-text numbers of the realistic request and its m-value.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
Rscript "$HERE/01_correction.R"
"$PYTHON" "$HERE/02_figure.py"
"$PYTHON" "$HERE/03_mvalue.py"
"$PYTHON" "$HERE/04_variables_table.py"
"$PYTHON" "$HERE/05_numbers.py"
