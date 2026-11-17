"""Pathways to Work: effect of program assignment on the 12-month overall summary index.

Index: the eight 12-month outcomes (excluding training), each z-scored against the
control group's mean and SD and signed so that higher = better; summed and re-standardized
to control-group SD units (Kling-Liebman-Katz style).
Estimation (CODEBOOK convention): OLS of the index on treatment + site indicators, adjusting
for the baseline covariates, HC2 robust standard errors, two-sided t-test of the treatment
coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

# outcome -> direction (+1 if higher is better, -1 if higher is worse)
COMPONENTS = {
    "employed_m12": 1,
    "hours_m12": 1,
    "earnings_m12": 1,
    "fin_security_m12": 1,
    "food_insecure_m12": -1,
    "debt_collections_m12": -1,
    "distress_m12": -1,
    "health_m12": 1,
}

COVARIATES = (
    "age + female + parent + C(education) + household_size"
    " + employed_baseline + earnings_baseline + months_unemployed_baseline"
)


def summary_index(df):
    control = df["treatment"] == 0
    total = 0
    for col, sign in COMPONENTS.items():
        total = total + sign * (df[col] - df.loc[control, col].mean()) / df.loc[control, col].std()
    return (total - total[control].mean()) / total[control].std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["index_m12"] = summary_index(df)
    fit = smf.ols(f"index_m12 ~ treatment + C(site) + {COVARIATES}", data=df).fit(cov_type="HC2")

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
