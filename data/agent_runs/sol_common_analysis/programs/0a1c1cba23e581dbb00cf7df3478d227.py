#!/usr/bin/env python3
"""Estimate the ITT effect on training enrollment at 12 months.

Dependencies: pandas, statsmodels.
Model: training_m12 ~ treatment + C(site), with an intercept.
Inference: HC2 robust standard error and a two-sided Student t test,
using the OLS residual degrees of freedom.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    columns = ["training_m12", "treatment", "site"]
    if data[columns].isna().any().any():
        raise ValueError("The selected analysis requires complete observations.")
    model = smf.ols(
        "training_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
