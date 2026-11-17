"""Rerun the single chosen analysis: effect of Pathways to Work on debt in
collections at 12 months (debt_collections_m12), OLS on treatment + site
indicators, HC2 standard errors, two-sided t-test on the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "debt_collections_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
