#!/usr/bin/env python3
"""Table 5 (known selection rules on a fixed menu) and its in-text numbers.

Inputs:  data/derived/07_immigration_menu/, data/agent_runs/issp_menu_{anti,pro}/ (originals.csv, replays.csv,
         inputs/), data/intermediate/07_immigration_menu/menu_full_data.csv
Outputs: paper/tabs/immigration_menu.tex, results/numbers/07_immigration_menu.json
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from paper_numbers import write_numbers  # noqa: E402

NAMES = {"fixed_reference": "Fixed model", "frontier": "Frontier", "client_pressure": "Client pressure"}
ARMS = {"anti": "Analyst expects a negative association", "pro": "Analyst expects a positive association"}


def main():
    derived = ROOT / "data/derived/07_immigration_menu"
    rep = {(r["arm"], r["workflow"]): r for r in csv.DictReader(open(derived / "replay.csv", encoding="utf-8"))}
    exact = {(r["arm"], r["workflow"]): r for r in csv.DictReader(open(derived / "exact.csv", encoding="utf-8"))}
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Workflow & Nominal $p$ & Tail & Body & Multiplier & Corrected $p$ & Exact $p$ \\"]
    for arm, title in ARMS.items():
        lines += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for w, name in NAMES.items():
            r, e = rep[(arm, w)], exact[(arm, w)]
            u = float(r["u"])
            lines.append(f"{name} & {u:.3f} & {r['x1']}/{r['b1']} & {r['x0']}/{r['b0']} & "
                         f"{float(r['multiplier']):.2f} ($\\geq$ {float(r['p_lower_one95']) / u:.2f}) & "
                         f"{float(r['corrected_p']):.3f} ($\\leq$ {float(r['p_upper_one95']):.3f}) & "
                         f"{float(e['exact_corrected_p']):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = ROOT / "paper/tabs/immigration_menu.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    runs = ROOT / "data/agent_runs"
    catalogue = json.loads((runs / "issp_menu_anti/inputs/CATALOGUE.json").read_text(encoding="utf-8"))["models"]
    menu = list(csv.DictReader(open(ROOT / "data/intermediate/07_immigration_menu/menu_full_data.csv",
                                    encoding="utf-8")))
    most_negative = min(menu, key=lambda m: float(m["hc3_t"]))["model_id"]
    named = {arm: sorted({m for f in (runs / f"issp_menu_{arm}/inputs/prompts").glob("*.txt")
                          for m in re.findall(r"AD\d{3}", f.read_text(encoding="utf-8"))}) for arm in ARMS}
    spec = {m["model_id"]: m for m in catalogue}
    originals = {arm: list(csv.DictReader(open(runs / f"issp_menu_{arm}/originals.csv", encoding="utf-8")))
                 for arm in ARMS}
    attempted = collections.Counter(
        (arm, r["workflow"], r["stratum"]) for arm in ARMS
        for r in csv.DictReader(open(runs / f"issp_menu_{arm}/replays.csv", encoding="utf-8")))
    per_stratum = sorted(set(attempted.values()))
    classified = sum(int(r["b1"]) + int(r["b0"]) for r in rep.values())
    followed = sum(int(r["followed_rule"]) for r in rep.values())
    anti = spec[named["anti"][0]]
    exact_sizes = {int(e["tail"]) + int(e["body"]) for e in exact.values()}
    named_quote = "The named specification for the anti-immigration analyst regresses support for government " \
                  "responsibility for the unemployed on a measure of the share of immigrants in the country without " \
                  "controls"
    negative_quote = "It is the most negative specification on the full dataset ($p = 0.026$)"
    lambdas = [float(rep[("anti", w)]["multiplier"]) for w in ("frontier", "client_pressure")]
    fixed = [float(rep[(a, "fixed_reference")]["multiplier"]) for a in ARMS]
    pro_u = [float(rep[("pro", w)]["u"]) for w in ("frontier", "client_pressure")]
    write_numbers("07_immigration_menu", [
        ("menu_specifications", "a menu of 108 OLS regressions", len(catalogue)),
        ("menu_outcomes", "country-level support for the six welfare-related outcomes in the ISSP data",
         len({m["outcome"] for m in catalogue})),
        ("menu_anti_named_spec", named_quote,
         f"{anti['model_id']}: {anti['outcome']} on {anti['exposure']}, controls {anti['control_set']}"),
        ("menu_anti_most_negative", negative_quote, most_negative),
        ("menu_anti_u", negative_quote, float(rep[("anti", "fixed_reference")]["u"])),
        ("menu_anti_originals", "so all three workflows report it as the ``original'' analysis",
         sum(o["selected_model_id"] == anti["model_id"] for o in originals["anti"])),
        ("menu_replays_per_stratum", "we replayed each workflow on 100-200 subsampled datasets per stratum",
         per_stratum),
        ("menu_bank_size", "an enlarged bank of $20{,}000$ subsampled datasets", sorted(exact_sizes)),
        ("menu_bank_size_note", "to all 20,000 subsamples",
         int(exact[("pro", "fixed_reference")]["tail"]) + int(exact[("pro", "fixed_reference")]["body"])),
        ("menu_followed_rule", "Every replay agent followed the rule it was assigned exactly",
         f"{followed}/{classified}"),
        ("menu_fixed_multiplier", "has an estimated multiplier of exactly $1$", fixed),
        ("menu_anti_lambdas", r"the other rules estimate $\lambda$s larger than $1$", [round(x, 4) for x in lambdas]),
        ("menu_pro_u_frontier", "($p=0.755$ and $0.919$)", round(pro_u[0], 6)),
        ("menu_pro_u_client", "($p=0.755$ and $0.919$)", round(pro_u[1], 6)),
    ])


if __name__ == "__main__":
    main()
