"""Rerun the single chosen analysis (no search).

Outcome:  food_insecure_m12 (household food insecurity at 12 months)
Sample:   participants with employed_baseline == 1
Model:    OLS of outcome on treatment + site indicators, HC2 robust SEs,
          two-sided t-test of the treatment coefficient (residual df).
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "food_insecure_m12"


def analyse(df):
    df = df[df["employed_baseline"] == 1]
    y = df[OUTCOME].to_numpy(float)
    sites = pd.get_dummies(df["site"]).to_numpy(float)  # full set of site dummies = intercept
    X = np.column_stack([df["treatment"].to_numpy(float), sites])
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    u = resid / np.sqrt(1.0 - h)  # HC2
    V = XtXi @ (X.T * u**2) @ X @ XtXi
    est, se = float(beta[0]), float(np.sqrt(V[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), n - k))
    return est, se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    est, se, p = analyse(pd.read_csv(a.data))
    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
