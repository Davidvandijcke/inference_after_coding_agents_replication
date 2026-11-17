"""Rerun the single chosen analysis: effect of Pathways to Work on self-rated
health at 6 months (health_m6) among participants at site 4.

OLS of health_m6 on treatment (plus site indicators if more than one site is
present), HC2 standard errors, two-sided t-test on the treatment coefficient,
per the CODEBOOK convention.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

SITE, OUTCOME = 4, "health_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d[d["site"] == SITE]
    y = d[OUTCOME].to_numpy(float)
    cols = [d["treatment"].to_numpy(float), np.ones(len(d))]
    for s in sorted(d["site"].unique())[1:]:  # no-op for a single site
        cols.append((d["site"] == s).to_numpy(float))
    X = np.column_stack(cols)

    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    u = resid / np.sqrt(1 - h)  # HC2
    V = XtXi @ (X.T * u**2) @ X @ XtXi
    est, se = beta[0], np.sqrt(V[0, 0])
    p = 2 * stats.t.sf(abs(est / se), len(y) - X.shape[1])

    with open(a.output, "w") as f:
        json.dump({"estimate": float(est), "std_error": float(se), "p_value": float(p)}, f, indent=2)


if __name__ == "__main__":
    main()
