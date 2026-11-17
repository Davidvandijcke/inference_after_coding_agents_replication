"""Rerun the single chosen analysis: effect of treatment on employment at 12 months.

OLS of employed_m12 on treatment + site indicators, HC2 standard errors,
two-sided t-test of the treatment coefficient (CODEBOOK.md convention).
Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    X = pd.get_dummies(d["site"].astype(int), prefix="site", drop_first=True).astype(float)
    X.insert(0, "treatment", d["treatment"].astype(float))
    X = sm.add_constant(X)
    r = sm.OLS(d["employed_m12"].astype(float), X).fit(cov_type="HC2", use_t=True)
    out = {"estimate": float(r.params["treatment"]),
           "std_error": float(r.bse["treatment"]),
           "p_value": float(r.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
