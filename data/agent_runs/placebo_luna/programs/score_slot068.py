#!/usr/bin/env python3
"""Score the module's effect on CO2-topic valuation responses.

Fits the block-adjusted OLS model for wtp_less_if_ineffective among respondents
assigned to the CO2 valuation topic, using HC2 robust standard errors.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subset = data.loc[data["wtp_topic"] == "co2"]
    model = smf.ols(
        "wtp_less_if_ineffective ~ treatment + C(block)", data=subset
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
