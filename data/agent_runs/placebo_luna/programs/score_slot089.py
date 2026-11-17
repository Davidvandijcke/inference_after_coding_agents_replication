#!/usr/bin/env python3
"""Score the treatment effect on fuel block beliefs about consumption impact."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data = data.loc[data["block"] == 1]
    model = smf.ols(
        "effect_below_one_to_one ~ treatment + C(block)", data=data
    ).fit(cov_type="HC2")
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
