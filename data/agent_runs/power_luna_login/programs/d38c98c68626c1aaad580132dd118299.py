#!/usr/bin/env python3
"""Score the preselected Pathways to Work subgroup outcome analysis.

Fits OLS among participants with some college (education == 3), regressing
12-month food insecurity on treatment assignment and site indicators. The
treatment coefficient uses HC2 heteroskedasticity-robust standard errors and
a two-sided t test, as specified in CODEBOOK.md.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"education", "food_insecure_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    subgroup = data.loc[data["education"] == 3]
    if subgroup.empty:
        raise ValueError("No participants with education == 3")

    result = smf.ols(
        "food_insecure_m12 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2")
    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
