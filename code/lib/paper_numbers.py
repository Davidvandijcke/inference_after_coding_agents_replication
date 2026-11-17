"""Writes results/numbers/<module>.json: the numbers the text states, each with the sentence that states it.

    write_numbers("08_card", [("card_lambda", "selection multiplier of 1.72", 1.7199)])

An entry is (id, quote, value). The quote is verbatim from paper/main.tex ("..." stands for omitted words);
its line is looked up there, and the script stops if a quote is not on exactly one line.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def locate(quote, lines):
    """Numbers of the lines that contain every "..."-separated piece of the quote, in order."""
    pieces = [" ".join(p.split()) for p in quote.split("...") if p.strip()]
    hits = []
    for number, line in enumerate(lines, 1):
        text, pos = " ".join(line.split()), 0
        for piece in pieces:
            pos = text.find(piece, pos)
            if pos < 0:
                break
            pos += len(piece)
        else:
            hits.append(number)
    return hits


def write_numbers(module, entries):
    lines = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8").split("\n")
    out, faults = [], []
    for id_, quote, value in entries:
        hits = locate(quote, lines)
        if len(hits) != 1:
            faults.append(f"{id_} (on {len(hits)} lines)")
        out.append({"id": id_, "main_tex_line": hits[0] if len(hits) == 1 else None, "quote": quote, "value": value})
    if len({e["id"] for e in out}) != len(out):
        faults.append("an id is used twice")
    target = ROOT / "results" / "numbers" / f"{module}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if faults:
        sys.exit(f"{module}: quotes not on exactly one line of paper/main.tex: " + ", ".join(faults))
    return out
