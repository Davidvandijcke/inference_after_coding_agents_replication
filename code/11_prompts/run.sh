#!/usr/bin/env bash
# Prompt displays of Appendix app:prompts: checks them against the prompt files, compiles prompts.tex and
# converts the glyphs to outlines (paper/figs/prompts.pdf). Needs latexmk, Ghostscript and pdftotext.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="$ROOT/paper/figs/prompts.pdf"
export SOURCE_DATE_EPOCH=0 FORCE_SOURCE_DATE=1
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "$HERE/01_check_displays.py"
tmp="$(mktemp -d)"
cp "$HERE/prompts.tex" "$tmp/"
if ! (cd "$tmp" && latexmk -pdf -interaction=nonstopmode -halt-on-error prompts.tex >latexmk.out 2>&1); then
  cat "$tmp/latexmk.out" >&2
  echo "prompts.tex failed to build; log kept in $tmp/prompts.log" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")"
gs -q -o "$OUT" -sDEVICE=pdfwrite -dNoOutputFonts -dOmitInfoDate -dOmitID -dOmitXMP "$tmp/prompts.pdf"
rm -rf "$tmp"
chars="$(pdftotext "$OUT" - | tr -d '[:space:]\f' | wc -c | tr -d ' ')"
[ "$chars" -eq 0 ] || { echo "prompts.pdf: $chars extractable characters remain" >&2; exit 1; }
