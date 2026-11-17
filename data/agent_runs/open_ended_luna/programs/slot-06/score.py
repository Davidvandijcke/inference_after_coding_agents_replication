#!/usr/bin/env python3
"""Score the exploratory under-35, six-month distress analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV with evaluation data")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["age"] < 35]
    result = smf.ols("distress_m6 ~ treatment + C(site)", data=subgroup).fit(
        cov_type="HC2", use_t=True
    )

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
