"""Rerun the chosen Pathways to Work analysis on another dataset.

Chosen test: effect of treatment on 6-month training enrollment (training_m6)
among participants employed at baseline (employed_baseline == 1), estimated by
OLS of the outcome on treatment and site indicators with HC2 standard errors
and a two-sided t-test, per the CODEBOOK.md convention.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    g = d[d["employed_baseline"] == 1]
    fit = smf.ols("training_m6 ~ treatment + C(site)", g).fit(cov_type="HC2")
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
