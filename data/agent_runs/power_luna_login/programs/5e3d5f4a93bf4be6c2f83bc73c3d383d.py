#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import json

import pandas as pd
from scipy.stats import t
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON output")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2"
    )
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=float(model.df_resid)))

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "estimate": estimate,
                "std_error": std_error,
                "p_value": p_value,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
