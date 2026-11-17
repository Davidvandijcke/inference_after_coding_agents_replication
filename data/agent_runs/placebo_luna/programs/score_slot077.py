#!/usr/bin/env python3
"""Estimate the block-adjusted treatment effect on capped ineffective-scenario WTP."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    analysis = data.dropna(subset=["wtp_ineffective_capped", "treatment", "block"])
    model = smf.ols(
        "wtp_ineffective_capped ~ treatment + C(block)", data=analysis
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
