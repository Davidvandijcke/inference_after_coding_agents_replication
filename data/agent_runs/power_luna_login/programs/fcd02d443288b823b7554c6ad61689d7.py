#!/usr/bin/env python3
"""Score the prespecified lead analysis for Pathways to Work.

Estimates the treatment effect on employment at 12 months using OLS with site
indicators and HC2 heteroskedasticity-robust standard errors.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2"
    )
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    p_value = float(2 * t.sf(abs(estimate / std_error), model.df_resid))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {
                "estimate": estimate,
                "std_error": std_error,
                "p_value": p_value,
            },
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
