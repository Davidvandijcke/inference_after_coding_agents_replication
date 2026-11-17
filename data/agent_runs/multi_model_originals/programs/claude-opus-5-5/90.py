"""Pre-specified test: Pathways to Work effect on 12-month earnings among
participants not employed at random assignment.

Convention (CODEBOOK.md): OLS of the outcome on `treatment` plus site
indicators, HC2 robust standard errors, two-sided t-test on `treatment`,
estimated within the subgroup employed_baseline == 0. No baseline covariates.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m12"
SUBGROUP_COL = "employed_baseline"
SUBGROUP_VALUE = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[df[SUBGROUP_COL] == SUBGROUP_VALUE]
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=df).fit(
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
