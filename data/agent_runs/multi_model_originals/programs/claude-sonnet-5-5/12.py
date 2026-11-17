"""Rerun the single chosen analysis: effect of Pathways to Work on 6-month
psychological distress (Kessler-6) among female participants.

OLS of distress_m6 on treatment + site indicators, HC2 standard errors,
two-sided t-test on the treatment coefficient (CODEBOOK.md convention).
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
    sub = d[d["female"] == 1]
    m = smf.ols("distress_m6 ~ treatment + C(site)", sub).fit(cov_type="HC2")
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
