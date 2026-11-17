#!/usr/bin/env bash
# Introduction figure (stylized illustration; no data inputs).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SOURCE_DATE_EPOCH=0 "$PYTHON" "$HERE/01_lens_figure.py"
