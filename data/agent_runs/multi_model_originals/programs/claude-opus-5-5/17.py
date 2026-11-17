#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen test (fixed in advance; no search is performed here):
    Does the program's effect on the 12-month financial security index
    (fin_security_m12) differ between participants who were employed at
    random assignment (employed_baseline == 1) and those who were not?

Estimate = (treatment effect among baseline-employed)
         - (treatment effect among baseline-not-employed),
where each subgroup effect follows the CODEBOOK convention: OLS of the outcome
on `treatment` plus site indicators within the subgroup, no further covariates.
The two subgroup regressions are stacked into one fully interacted OLS model
(every coefficient, site indicators included, is allowed to differ by baseline
employment), so the coefficient on treatment x employed_baseline is exactly the
difference. Standard error: HC2. Test: two-sided t-test of a zero coefficient
(residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "fin_security_m12"
MODERATOR = "employed_baseline"
REQUIRED = ["site", "treatment", MODERATOR, OUTCOME]


def build_design(df):
    t = df["treatment"].astype(float).to_numpy()
    g = df[MODERATOR].astype(float).to_numpy()
    cols = {
        "const": np.ones(len(df)),
        MODERATOR: g,
        "treatment": t,
        "treatment_x_" + MODERATOR: t * g,
    }
    # Site indicators estimated separately within each baseline-employment group.
    for level in (0, 1):
        in_group = g == level
        sites = sorted(df.loc[in_group, "site"].unique())
        for s in sites[1:]:
            cols[f"site{s}_grp{level}"] = ((df["site"] == s).to_numpy() & in_group).astype(float)
    return pd.DataFrame(cols, index=df.index)


def run(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df[REQUIRED].dropna()
    for c in ("treatment", MODERATOR):
        if not set(df[c].unique()) <= {0, 1}:
            raise ValueError(f"column {c} must be coded 0/1")
    for level in (0, 1):
        sub = df[df[MODERATOR] == level]
        if sub["treatment"].nunique() < 2:
            raise ValueError(f"{MODERATOR}=={level} subgroup lacks both treatment arms")

    X = build_design(df)
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError("design matrix is rank deficient")
    fit = sm.OLS(df[OUTCOME].astype(float), X).fit(cov_type="HC2", use_t=True)
    term = "treatment_x_" + MODERATOR
    return {
        "estimate": float(fit.params[term]),
        "std_error": float(fit.bse[term]),
        "p_value": float(fit.pvalues[term]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="CSV with the codebook's columns")
    ap.add_argument("--output", required=True, help="path for the JSON result")
    args = ap.parse_args()
    result = run(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
