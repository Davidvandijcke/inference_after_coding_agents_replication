#!/usr/bin/env python3
"""Estimate the year-2 program effect on job satisfaction among age 50+."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["age_group"] == 3].copy()
    analysis = subgroup.dropna(
        subset=["satisfied_job_y2", "satisfied_job_base", "treatment", "block"]
    )
    model = smf.ols(
        "satisfied_job_y2 ~ treatment + C(block) + satisfied_job_base",
        data=analysis,
    ).fit(cov_type="HC2")

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
