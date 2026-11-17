#!/usr/bin/env bash
# Table 5 (known selection rules on a fixed menu) and its in-text numbers.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
"$PYTHON" "$HERE/01_exact.py"
Rscript "$HERE/02_replay.R"
"$PYTHON" "$HERE/03_table.py"
