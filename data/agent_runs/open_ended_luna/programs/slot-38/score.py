#!/usr/bin/env python3
"""Score the exploratory parent-status interaction in 6-month distress."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the treatment-by-parent interaction for 6-month Kessler-6 "
            "distress, allowing site effects to differ by parent status, with "
            "HC2 standard errors."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "parent", "distress_m6"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))
    if data[list(required)].isna().any().any():
        parser.error("required analysis columns must not contain missing values")

    model = smf.ols(
        "distress_m6 ~ treatment * parent + C(site) * parent", data=data
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:parent"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
