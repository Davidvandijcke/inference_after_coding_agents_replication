#!/usr/bin/env python3
"""Score the treatment-effect contrast for CO2 versus other WTP topics.

The estimate is the coefficient on treatment * co2 in an OLS model of
wtp_ineffective_capped with block indicators. HC2 standard errors and a
two-sided t-test are used, following CODEBOOK.md.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data["co2"] = (data["wtp_topic"] == "co2").astype(int)

    model = smf.ols(
        'wtp_ineffective_capped ~ treatment * co2 + C(block)', data=data
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:co2"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
