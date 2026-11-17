#!/usr/bin/env python
"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month
self-rated health (health_m12), per the CODEBOOK convention (OLS of outcome on
treatment + site indicators, HC2 standard errors, two-sided t-test on treatment).
No outcome/subgroup search is performed."""
import argparse, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUTCOME = "health_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    X = pd.concat([d[["treatment"]].astype(float),
                   pd.get_dummies(d["site"], prefix="site", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    m = sm.OLS(d[OUTCOME].astype(float), X).fit(cov_type="HC2")
    est, se = float(m.params["treatment"]), float(m.bse["treatment"])
    # two-sided t-test; df = residual df (statsmodels uses normal for robust covariance by default)
    p = float(2 * stats.t.sf(abs(est / se), m.df_resid))
    json.dump({"estimate": est, "std_error": se, "p_value": p}, open(a.output, "w"), indent=2)


if __name__ == "__main__":
    main()
