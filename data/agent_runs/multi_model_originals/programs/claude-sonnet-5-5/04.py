"""Chosen analysis: effect of Pathways to Work on employment at 12 months
(employed_m12), full sample. OLS of the outcome on treatment and site
indicators, HC2 standard errors, two-sided t-test on the treatment coefficient
(CODEBOOK.md convention). No search is performed here.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    m = smf.ols("employed_m12 ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    res = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
