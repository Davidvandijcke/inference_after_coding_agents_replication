#!/usr/bin/env bash
# Rebuilds every table and figure into paper/tabs and paper/figs and compiles paper/main.tex.
# Run "bash environment/setup.sh" once first. Logs go to logs/. No step calls a language model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export SOURCE_DATE_EPOCH=0
export PYTHONDONTWRITEBYTECODE=1
LENS="${PYTHON_LENS:-$ROOT/.venv-lens/bin/python}"
mkdir -p "$ROOT/logs"

for tool in Rscript latexmk gs pdftotext; do
  command -v "$tool" > /dev/null 2>&1 || { echo "run_all.sh: $tool is not on the PATH (see the README's software requirements)" >&2; exit 1; }
done
for python in "$PYTHON" "$LENS"; do
  command -v "$python" > /dev/null 2>&1 || { echo "run_all.sh: $python not found; run 'bash environment/setup.sh' first, or set PYTHON and PYTHON_LENS" >&2; exit 1; }
done

MODULES=(
  00_lens_figure
  01_fixed_design
  03_placebo          # before 02, whose Figure 3(a) uses its results
  02_open_ended
  04_power
  05_immigration_personas
  06_immigration_request
  07_immigration_menu
  08_card
  09_card_agents
  10_self_correction
  11_prompts
)

for module in "${MODULES[@]}"; do
  start=$SECONDS
  printf '%-26s' "$module"
  python="$PYTHON"
  [ "$module" = 00_lens_figure ] && python="$LENS"  # drawn with matplotlib 3.11.2
  if ! PYTHON="$python" bash "$ROOT/code/$module/run.sh" > "$ROOT/logs/$module.log" 2>&1; then
    printf 'FAILED after %ss; the last lines of logs/%s.log:\n' "$((SECONDS - start))" "$module"
    tail -n 15 "$ROOT/logs/$module.log"
    exit 1
  fi
  printf 'done in %ss\n' "$((SECONDS - start))"
done

printf '%-26s' "paper"
if ! (cd "$ROOT/paper" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex) \
  > "$ROOT/logs/paper.log" 2>&1; then
  echo "FAILED; the last lines of logs/paper.log:"
  tail -n 15 "$ROOT/logs/paper.log"
  exit 1
fi
echo "done: paper/main.pdf"
