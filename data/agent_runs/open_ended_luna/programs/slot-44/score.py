#!/usr/bin/env python3
"""Score the exploratory parent subgroup food insecurity analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to a CSV with the study columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"parent", "food_insecure_m12", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    parents = data.loc[data["parent"] == 1,
                       ["food_insecure_m12", "treatment", "site"]]
    if parents.empty:
        parser.error("data contains no participants with parent == 1")

    # Site fixed effects reflect the within-site randomization. HC2 robust
    # standard errors and a two-sided t-test follow the codebook convention.
    fit = smf.ols(
        "food_insecure_m12 ~ treatment + C(site)", data=parents
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
