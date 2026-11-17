"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test: among participants aged 40+ at random assignment, the effect of
assignment to the program on employment at the 6-month follow-up
(employed_m6). OLS of employed_m6 on treatment, site indicators and the
baseline covariates; HC2 standard errors; two-sided t-test of the treatment
coefficient (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "employed_m6 ~ treatment + C(site)"
    " + age + female + parent + C(education) + household_size"
    " + employed_baseline + earnings_baseline + months_unemployed_baseline"
)
AGE_CUTOFF = 40


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    d = pd.read_csv(args.data)
    sub = d[d["age"] >= AGE_CUTOFF]
    fit = smf.ols(FORMULA, data=sub).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
