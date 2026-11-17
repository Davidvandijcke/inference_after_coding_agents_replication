#!/usr/bin/env python3
"""Table 6: the variables of the Card-Krueger data the agents received, grouped by role. Stops unless the table lists
exactly the variables of the agents' codebook.

Input:   data/agent_runs/card_luna/packet/CODEBOOK.md
Output:  paper/tabs/card_variables.tex
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARD = [
    ("Store", [
        (["store_id", "sheet"], "Store identifier (state and survey sheet number)", ""),
        (["chain"], "Burger King, KFC, Roy Rogers, Wendy\\textquoteright s", "1--4"),
        (["co_owned"], "Company-owned", "0/1"),
        (["state"], "New Jersey (1) or Pennsylvania (0)", "0/1"),
        (["southj", "centralj", "northj", "shore"], "Region of New Jersey; shore location", "0/1"),
        (["pa1", "pa2"], "Region of Pennsylvania", "0/1"),
    ]),
    ("Employment (wave 1 / wave 2)", [
        (["empft / empft2"], "Full-time employees, excluding managers", "count"),
        (["emppt / emppt2"], "Part-time employees, excluding managers", "count"),
        (["nmgrs / nmgrs2"], "Managers and assistant managers", "count"),
    ]),
    ("Wages and benefits (wave 1 / wave 2)", [
        (["wage_st / wage_st2"], "Starting wage", "\\$ per hour"),
        (["inctime / inctime2"], "Months to the usual first raise", "months"),
        (["firstinc / firstin2"], "Usual first raise", "\\$ per hour"),
        (["bonus / special2"], "Cash bounty (wave 1) or special program (wave 2) for new workers", "0/1"),
        (["pctaff"], "Share of employees affected by the new minimum wage (wave 1)", "\\%"),
        (["meals / meals2"], "Meal benefit: none, free, reduced-price, both", "0--3"),
    ]),
    ("Prices and operations (wave 1 / wave 2)", [
        (["psoda / psoda2"], "Price of a medium soda, including tax", "\\$"),
        (["pfry / pfry2"], "Price of small fries, including tax", "\\$"),
        (["pentree / pentree2"], "Price of an entree, including tax", "\\$"),
        (["open / open2r"], "Opening hour", "hour"),
        (["hrsopen / hrsopen2"], "Hours open per day", "hours"),
        (["nregs / nregs2"], "Cash registers", "count"),
        (["nregs11 / nregs112"], "Cash registers open at 11 am", "count"),
    ]),
    ("Survey (wave 1 / wave 2)", [
        (["ncalls / ncalls2"], "Call-backs before contact", "count"),
        (["type2"], "Wave-2 interview by phone (1) or in person (2)", "1--2"),
        (["status2"], "Wave-2 status: refused, answered, or closed (renovation, permanently, highway construction, "
                      "mall fire)", "0--5"),
        (["date2"], "Wave-2 interview date", "MMDDYY"),
    ]),
]


def main():
    text = (ROOT / 'data/agent_runs/card_luna/packet/CODEBOOK.md').read_text(encoding='utf-8')
    names = set(re.findall(r"^\| ([a-z_0-9]+) \|", text, flags=re.M)) | {"store_id"}
    listed = [n for _, rows in CARD for vs, _, _ in rows for v in vs for n in v.split(" / ")]
    if "store_id" not in text:
        sys.exit("the agents' codebook does not describe store_id")
    if set(listed) != names or len(listed) != len(set(listed)):
        sys.exit(f"the table and the agents' codebook list different variables: {sorted(set(listed) ^ names)}")
    tex_names = lambda vs: ", ".join("\\texttt{" + v.replace("_", "\\_") + "}" for v in vs)
    # \raggedright only in the first two columns, so the third column's \\ still ends the row
    out =[r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}p{0.26\linewidth}p{0.52\linewidth}p{0.16\linewidth}@{}}",
           r"\toprule", r"Variable & Description & Coding \\"]
    for title, rows in CARD:
        out += [r"\midrule", rf"\multicolumn{{3}}{{@{{}}p{{0.96\linewidth}}}}{{\textit{{{title}}}}} \\"]
        out += [rf"\raggedright {tex_names(vs)} & \raggedright {desc} & {code} \\" for vs, desc, code in rows]
    path = ROOT / 'paper/tabs/card_variables.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding='utf-8')


if __name__ == '__main__':
    main()
