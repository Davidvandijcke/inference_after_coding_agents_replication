"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen test (exploratory, selected from a subgroup search):
    Among participants aged 45 or older at random assignment, the effect of
    assignment to the program on self-rated health at 12 months (health_m12).
    OLS of health_m12 on treatment and site indicators, HC2 robust standard
    errors, two-sided t-test of a zero coefficient on treatment.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

AGE_CUTOFF = 45
OUTCOME = "health_m12"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    d = pd.read_csv(args.data)
    sub = d[d["age"] >= AGE_CUTOFF]

    # Site indicators because randomization was within site; drop them only if
    # the subgroup happens to contain a single site.
    formula = f"{OUTCOME} ~ treatment" + (" + C(site)" if sub["site"].nunique() > 1 else "")
    fit = smf.ols(formula, data=sub).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
