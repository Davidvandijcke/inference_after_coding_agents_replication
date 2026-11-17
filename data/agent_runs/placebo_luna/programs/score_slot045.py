#!/usr/bin/env python3
"""Score the year-2 treatment effect on current smoking.

Fits the codebook's block-adjusted OLS model, controlling for baseline current
smoking, and reports the HC2 standard error and two-sided t-test p-value.
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

    outcome = "current_smoker_y2"
    baseline = "current_smoker_base"
    frame = pd.read_csv(args.data)
    required = {outcome, baseline, "treatment", "block"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    analysis = frame[[outcome, baseline, "treatment", "block"]].dropna()
    if analysis.empty:
        raise ValueError("No complete observations for the chosen analysis")

    fit = smf.ols(
        f"{outcome} ~ treatment + C(block) + {baseline}", data=analysis
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
