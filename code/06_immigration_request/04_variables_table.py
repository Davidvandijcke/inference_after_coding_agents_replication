#!/usr/bin/env python3
"""Table tab:issp_variables: the variables of the ISSP file the agents received, condensed from their codebook.

Input:  data/agent_runs/issp_request_luna/inputs/CODEBOOK.md
Output: paper/tabs/issp_variables.tex
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GROUPS = [
    ("Outcomes: ``should it be the government\\textquoteright s responsibility to \\ldots''; "
     "1 definitely should be, 2/3 probably should be, 1/3 probably should not be, 0 definitely should not be", [
        (["jobs"], "provide a job for everyone who wants one", "0--1"),
        (["healthcare"], "provide health care for the sick", "0--1"),
        (["old_age"], "provide a decent standard of living for the old", "0--1"),
        (["unemployed"], "provide a decent standard of living for the unemployed", "0--1"),
        (["income_diff"], "reduce income differences between the rich and the poor", "0--1"),
        (["housing"], "provide decent housing for those who can\\textquoteright t afford it (not asked in 1985)", "0--1"),
    ]),
    ("Immigration (country, module year)", [
        (["migstock"], "Immigrant stock: World Bank where available, otherwise UN", "\\% of population"),
        (["migstock_wb", "migstock_un", "migstock_oecd"],
         "Immigrant stock from the World Bank (1985--2006), the UN (1990--2016) and the OECD (members only)",
         "\\% of population"),
        (["mignet_un"], "Net migration rate, UN (none for 2016)", "per 1,000 per year"),
        (["migflow_pct"], "The same rate in percent", "\\% per year"),
    ]),
    ("Respondent", [
        (["age"], "Age", "years"),
        (["female"], "Female", "0/1"),
        (["employed"], "In paid work", "0/1"),
        (["educ"], "Highest completed education, harmonized (1996--2016)", "0--5"),
        (["topbot"], "Self-placement from the bottom to the top of society (2006--2016)", "1--10"),
    ]),
    ("Country controls (module year)", [
        (["gdp_wb", "log_gdp"], "GDP per capita and its logarithm", "current US\\$"),
        (["pop_wb"], "Population", "persons"),
        (["ginid_solt"], "Gini index of disposable household income (SWIID)", "0--1"),
        (["gini_wb"], "Gini index (World Bank; sparse)", "0--100"),
        (["wdi_unempilo"], "Unemployment rate (ILO; from 1991)", "\\% of labor force"),
        (["socx_oecd"], "Public social expenditure (OECD members)", "\\% of GDP"),
        (["al_ethnic"], "Ethnic fractionalization", "0--1"),
    ]),
    ("Identifiers", [
        (["wave", "year"], "Survey module: 1985, 1990, 1996, 2006, 2016", ""),
        (["iso_country", "cntry_wave"], "Country (ISO code) and country-wave", ""),
    ]),
]


def codebook_names(text):
    """Backticked first cells of the codebook's variable tables, from the Variables section on."""
    body = text[text.index("## Variables"):]
    return set(re.findall(r"^\| `([a-z_0-9]+)` \|", body, flags=re.M))


def tex_names(vs):
    return ", ".join("\\texttt{" + v.replace("_", "\\_") + "}" for v in vs)


def main():
    names = codebook_names((ROOT / "data/agent_runs/issp_request_luna/inputs/CODEBOOK.md").read_text(encoding="utf-8"))
    listed = [v for _, rows in GROUPS for vs, _, _ in rows for v in vs]
    assert set(listed) == names and len(listed) == len(set(listed)), (sorted(set(listed) ^ names))
    # \raggedright only in the first two columns, so the row-ending \\ in the third keeps its tabular meaning
    out = [r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}p{0.26\linewidth}p{0.52\linewidth}p{0.16\linewidth}@{}}", r"\toprule",
           r"Variable & Description & Coding \\"]
    for title, rows in GROUPS:
        out += [r"\midrule", rf"\multicolumn{{3}}{{@{{}}p{{0.96\linewidth}}}}{{\textit{{{title}}}}} \\"]
        out += [rf"\raggedright {tex_names(vs)} & \raggedright {desc} & {code} \\" for vs, desc, code in rows]
    path = ROOT / "paper/tabs/issp_variables.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
