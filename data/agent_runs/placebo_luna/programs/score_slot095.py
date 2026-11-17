#!/usr/bin/env python3
"""Score the household-energy block treatment effect on capped effective WTP."""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    sample = data.loc[data["block"] == 4, ["block", "treatment", "wtp_effective_capped"]].dropna()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("Household-energy block (block=4) must contain both treatment arms")

    model = smf.ols("wtp_effective_capped ~ treatment + C(block)", data=sample).fit(cov_type="HC2")
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
