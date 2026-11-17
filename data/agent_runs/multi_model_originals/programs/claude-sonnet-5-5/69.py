#!/usr/bin/env python
"""Re-run the single pre-chosen analysis from the Pathways to Work exploration.

Estimand: difference in the treatment effect on 12-month financial security
(fin_security_m12) between participants unemployed >= 6 months at baseline and
those unemployed < 6 months, i.e. the coefficient on treatment x long_unemp in

    fin_security_m12 ~ treatment + long_unemp + treatment:long_unemp + site dummies

Convention (CODEBOOK.md): OLS, site indicators, HC2 robust SEs, two-sided t-test
of a zero coefficient (df = n - k). No search is performed here.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "fin_security_m12"
THRESHOLD_MONTHS = 6


def hc2_coef(y, X, k):
    """OLS coefficient k with HC2 standard error and t-test p-value."""
    n, p = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    lev = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    a = xtx_inv[k] @ X.T                      # influence weights for coefficient k
    var = np.sum(a ** 2 * resid ** 2 / (1.0 - lev))
    se = float(np.sqrt(var))
    rank = np.linalg.matrix_rank(X)
    t = beta[k] / se
    p_val = float(2 * stats.t.sf(abs(t), n - rank))
    return float(beta[k]), se, p_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    treat = d["treatment"].to_numpy(float)
    lu = (d["months_unemployed_baseline"] >= THRESHOLD_MONTHS).to_numpy(float)
    sites = sorted(d["site"].unique())
    cols = [np.ones(len(d)), treat, lu, treat * lu]
    cols += [(d["site"] == s).to_numpy(float) for s in sites[1:]]
    X = np.column_stack(cols)
    y = d[OUTCOME].to_numpy(float)

    est, se, p = hc2_coef(y, X, k=3)
    with open(args.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
