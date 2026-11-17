"""Pathways to Work: covariate-adjusted program effect on usual weekly hours at 6 months.

Prespecified single test (no search): OLS of hours_m6 on treatment, site
indicators, and the codebook baseline covariates; HC2 robust SEs; two-sided
t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "hours_m6"
COVARIATES = [
    "age", "female", "parent", "education", "household_size",
    "employed_baseline", "earnings_baseline", "months_unemployed_baseline",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    cols = [OUTCOME, "treatment", "site"] + COVARIATES
    d = d[cols].dropna()

    formula = f"{OUTCOME} ~ treatment + C(site) + " + " + ".join(COVARIATES)
    fit = smf.ols(formula, data=d).fit(cov_type="HC2", use_t=True)
    ci = fit.conf_int().loc["treatment"]

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "n": int(fit.nobs),
        "outcome": OUTCOME,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
