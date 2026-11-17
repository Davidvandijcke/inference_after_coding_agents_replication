#!/usr/bin/env python3
"""Exact corrected p-value of each fixed-menu workflow, from its rule's pick on all 20,000 bank subsets.

Inputs:  data/intermediate/07_immigration_menu/bank_{anti,pro}.csv.gz, data/agent_runs/issp_menu_{anti,pro}/design.json
Output:  data/derived/07_immigration_menu/exact.csv
"""
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ("fixed_reference", "frontier", "client_pressure")


def main():
    rows = []
    for arm in ("anti", "pro"):
        design = json.loads((ROOT / f"data/agent_runs/issp_menu_{arm}/design.json").read_text(encoding="utf-8"))
        bank = list(csv.DictReader(gzip.open(ROOT / f"data/intermediate/07_immigration_menu/bank_{arm}.csv.gz", "rt",
                                                encoding="utf-8")))
        assert len(bank) == 20000
        for w in WORKFLOWS:
            target, u = design[w]["target"], design[w]["u"]
            if arm == "anti":
                p_col, s_col = "p_target", "stratum"
            else:
                p_col, s_col = ("p_second", "stratum_second") if w == "client_pressure" else ("p_primary", "stratum_primary")
            assert all((float(b[p_col]) <= u) == (b[s_col] == "tail") for b in bank)
            pick = {"fixed_reference": lambda b: True, "frontier": lambda b: b["frontier_pick"] == target,
                    "client_pressure": lambda b: b["client_pick"] == target}[w]
            tail = [pick(b) for b in bank if b[s_col] == "tail"]
            body = [pick(b) for b in bank if b[s_col] == "body"]
            a1, a0 = sum(tail) / len(tail), sum(body) / len(body)
            h = u * a1 / (u * a1 + (1 - u) * a0)
            rows.append({"arm": arm, "workflow": w, "target": target, "u": repr(u), "tail": len(tail),
                         "tail_selected": sum(tail), "body": len(body), "body_selected": sum(body),
                         "exact_corrected_p": repr(h), "exact_multiplier": repr(h / u)})
    out = ROOT / "data/derived/07_immigration_menu/exact.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
