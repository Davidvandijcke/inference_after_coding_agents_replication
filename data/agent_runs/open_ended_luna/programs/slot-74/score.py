#!/usr/bin/env python3
"""Score the parent-status employment heterogeneity analysis.

Fits OLS for 12-month employment with treatment, parent status, their
interaction, and site indicators. Reports the treatment-by-parent coefficient
with an HC2 standard error and a two-sided t-test.

Requires pandas and statsmodels.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    columns = ["site", "treatment", "parent", "employed_m12"]
    data = pd.read_csv(args.data, usecols=columns)
    if data[columns].isna().any().any():
        raise ValueError("Analysis columns must not contain missing values")

    result = smf.ols(
        "employed_m12 ~ treatment * parent + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment:parent"]),
        "std_error": float(result.bse["treatment:parent"]),
        "p_value": float(result.pvalues["treatment:parent"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
