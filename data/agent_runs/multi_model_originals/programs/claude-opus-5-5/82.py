"""Pathways to Work: pre-specified rerun of the chosen exploratory finding.

Test: effect of assignment to the program on employment at 12 months
(`employed_m12`) among participants unemployed for 6+ months at random
assignment (`months_unemployed_baseline >= 6`). OLS of the outcome on
`treatment` and site indicators, HC2 robust standard errors, two-sided t-test
of the `treatment` coefficient (residual degrees of freedom).

Usage: python score.py --data <path.csv> --output <path.json>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
SUBGROUP_MIN_MONTHS_UNEMPLOYED = 6


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    sub = d[d["months_unemployed_baseline"] >= SUBGROUP_MIN_MONTHS_UNEMPLOYED]
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
