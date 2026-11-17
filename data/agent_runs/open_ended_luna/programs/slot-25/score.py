#!/usr/bin/env python3
"""Score the chosen exploratory subgroup analysis for Pathways to Work."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the site-adjusted treatment effect on 12-month food "
            "insecurity among participants with less than high school education."
        )
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["education"] == 1]
    fit = smf.ols(
        "food_insecure_m12 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2")

    estimate = float(fit.params["treatment"])
    std_error = float(fit.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * student_t.sf(abs(statistic), df=fit.df_resid))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "estimate": estimate,
                "std_error": std_error,
                "p_value": p_value,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
