"""Pathways to Work: pre-specified rerun of the chosen finding.

Effect of random assignment to the program on having a debt in collections
at the 12-month follow-up (debt_collections_m12), full sample.
Per the CODEBOOK convention: OLS of the outcome on `treatment` plus site
indicators, HC2 robust standard errors, two-sided t-test of the treatment
coefficient (t distribution with residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "debt_collections_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data, usecols=[OUTCOME, "treatment", "site"]).dropna()
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=d).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
