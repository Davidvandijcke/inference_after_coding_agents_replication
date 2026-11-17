"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Finding: effect of assignment to the program on food insecurity (mean of the
6- and 12-month indicators) among participants not employed at random
assignment (employed_baseline == 0).

Estimation follows the CODEBOOK convention: OLS of the outcome on `treatment`
and site indicators, HC2 standard errors, two-sided t-test on `treatment`.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    d = pd.read_csv(args.data)
    sub = d.loc[d["employed_baseline"] == 0].copy()
    sub["food_insecure_avg"] = (sub["food_insecure_m6"] + sub["food_insecure_m12"]) / 2

    fit = smf.ols("food_insecure_avg ~ treatment + C(site)", data=sub).fit(cov_type="HC2")
    # statsmodels uses the normal distribution for robust covariances by default;
    # the convention calls for a t-test, so use the residual degrees of freedom.
    test = fit.t_test("treatment = 0", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(test.pvalue),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
