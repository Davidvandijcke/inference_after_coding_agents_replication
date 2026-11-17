#!/usr/bin/env python3
"""Score the selected Pathways to Work education-group comparison.

Analysis: OLS of employed_m12 on treatment, education group (education <= 2),
their interaction, and site indicators interacted with education group. This
allows separate site intercepts within each education group. The reported
coefficient is the difference in treatment effects (education <= 2 minus
education >= 3), with HC2 standard errors and a two-sided t-test. Requires
pandas and statsmodels.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = {"education", "employed_m12", "treatment", "site"}
    missing = sorted(needed - set(data.columns))
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    data = data.copy()
    data["low_education"] = (data["education"] <= 2).astype(int)
    if data["low_education"].nunique() < 2:
        parser.error("both education groups (education <= 2 and >= 3) are required")

    result = smf.ols(
        "employed_m12 ~ treatment * low_education + C(site) * low_education",
        data=data,
    ).fit(cov_type="HC2")
    term = "treatment:low_education"
    payload = {
        "estimate": float(result.params[term]),
        "std_error": float(result.bse[term]),
        "p_value": float(result.pvalues[term]),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
