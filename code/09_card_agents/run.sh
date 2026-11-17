#!/usr/bin/env bash
# Table 8 and its in-text numbers. Reads module 08's shipped records and banks.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
"$PY" "$HERE/01_counts.py"
Rscript "$HERE/02_score_bank.R"
Rscript "$HERE/03_replay.R"
"$PY" "$HERE/04_keyword_rates.py"
"$PY" "$HERE/05_mask_packet.py"
"$PY" "$HERE/06_table.py"
