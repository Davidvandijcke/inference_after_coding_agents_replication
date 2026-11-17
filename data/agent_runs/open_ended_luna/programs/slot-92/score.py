#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis on a CSV dataset.

The analysis is the site-adjusted treatment effect on 6-month food insecurity
among participants coded female=0, using HC2 robust standard errors and a
two-sided t-test as specified in CODEBOOK.md.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "female", "food_insecure_m6"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

    subgroup = data.loc[data["female"] == 0]
    if subgroup.empty:
        raise ValueError("Input CSV has no participants coded female=0")

    result = smf.ols(
        "food_insecure_m6 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
