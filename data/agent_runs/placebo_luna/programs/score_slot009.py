#!/usr/bin/env python3
"""Run the prespecified selected analysis on a supplied evaluation CSV."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"block", "treatment", "very_productive_y1"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Analyze observed one-year outcomes, adjusting for randomization blocks.
    analysis = data[["very_productive_y1", "treatment", "block"]].dropna()
    block_indicators = pd.get_dummies(
        analysis["block"], prefix="block", drop_first=True, dtype=float
    )
    design = pd.concat(
        [analysis[["treatment"]].astype(float), block_indicators], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    result = sm.OLS(analysis["very_productive_y1"], design).fit(cov_type="HC2")

    estimate = float(result.params["treatment"])
    std_error = float(result.bse["treatment"])
    t_stat = estimate / std_error
    p_value = float(2 * student_t.sf(abs(t_stat), result.df_resid))

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
