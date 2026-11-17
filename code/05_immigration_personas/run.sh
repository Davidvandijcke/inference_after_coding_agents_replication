#!/usr/bin/env bash
# Tables 2 and 4 (persona design of Miao et al.) and their in-text numbers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "$HERE/01_read_sfig3c.py"
"$PYTHON" "$HERE/02_persona_tables.py"
