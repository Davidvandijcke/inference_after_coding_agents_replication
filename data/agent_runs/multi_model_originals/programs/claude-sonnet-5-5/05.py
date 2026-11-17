#!/usr/bin/env python3
"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month
financial security (fin_security_m12) among participants with less than a high
school education (education == 1). OLS on treatment + site indicators,
HC2 robust SE, two-sided t-test on the treatment coefficient."""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    s = d[d["education"] == 1]
    m = smf.ols("fin_security_m12 ~ treatment + C(site)", s).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
