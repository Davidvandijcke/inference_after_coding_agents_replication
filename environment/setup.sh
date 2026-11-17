#!/usr/bin/env bash
# Creates the two Python environments and installs the R packages run_all.sh needs.
#   .venv       every module except 00 (requirements.txt)
#   .venv-lens  module 00, the introduction figure (requirements-lens.txt: matplotlib 3.11.2)
# Set PYTHON_BIN to choose the interpreter (tested with Python 3.14.2; at least 3.12 is required).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "Python 3.12 or later is required (set PYTHON_BIN); found " + sys.version.split()[0])'
for pair in ".venv:requirements.txt" ".venv-lens:requirements-lens.txt"; do
  env="${pair%%:*}"; req="${pair#*:}"
  "$PYTHON_BIN" -m venv --clear "$ROOT/$env"
  "$ROOT/$env/bin/python" -m pip install --quiet --disable-pip-version-check -r "$ROOT/environment/$req"
  "$ROOT/$env/bin/python" -c "import sys, numpy, matplotlib; print('$env: Python', sys.version.split()[0], 'numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"
done
Rscript "$ROOT/environment/install_r_packages.R"
