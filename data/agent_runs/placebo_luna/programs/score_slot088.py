#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on desired work under partner flexibility."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


OUTCOME = "if_partner_flexible_post"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset with the study columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    analysis = data[[OUTCOME, "treatment", "block"]].dropna()
    model = smf.ols(f'{OUTCOME} ~ treatment + C(block)', data=analysis).fit(cov_type="HC2")

    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=model.df_resid))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
