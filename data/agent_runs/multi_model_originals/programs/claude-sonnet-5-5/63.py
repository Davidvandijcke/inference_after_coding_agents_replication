#!/usr/bin/env python3
"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month
training enrollment (training_m12) among male participants (female == 0).

Convention (CODEBOOK.md): OLS of the outcome on `treatment` plus site
indicators, HC2 robust standard errors, two-sided t-test on `treatment`.
No searching over outcomes, subgroups or follow-up points is done here.
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
    d = d[d["female"] == 0]
    m = smf.ols("training_m12 ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
