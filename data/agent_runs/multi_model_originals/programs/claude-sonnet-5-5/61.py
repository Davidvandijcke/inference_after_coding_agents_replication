"""Rerun the chosen Pathways to Work analysis on another dataset.

Chosen analysis: effect of treatment on employment at 12 months
(employed_m12), OLS with site indicators, HC2 robust SE, two-sided t-test.
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
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
