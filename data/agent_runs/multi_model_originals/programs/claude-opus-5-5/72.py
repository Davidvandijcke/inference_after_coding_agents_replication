"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (selected after exploratory analysis of the original data):
  Among parents (parent == 1), the effect of random assignment to the program on
  enrollment in education or job training at the 12-month follow-up
  (training_m12). OLS of training_m12 on treatment plus site indicators,
  HC2 robust standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    sub = df.loc[df["parent"] == 1, ["training_m12", "treatment", "site"]].dropna()

    fit = smf.ols("training_m12 ~ treatment + C(site)", data=sub).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
