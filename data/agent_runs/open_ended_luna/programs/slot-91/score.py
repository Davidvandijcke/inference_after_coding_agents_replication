#!/usr/bin/env python3
"""Score the selected exploratory finding: six-month earnings in Site 3."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the Site 3 treatment effect on six-month earnings."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "earnings_m6"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("Missing required columns: " + ", ".join(sorted(missing)))

    site_data = data.loc[data["site"] == 3]
    if site_data.empty:
        parser.error("The input data contains no participants from Site 3")
    if site_data["treatment"].nunique() != 2:
        parser.error("Site 3 must contain both treatment and control participants")

    # Within Site 3, this is OLS of earnings on treatment; HC2 standard errors
    # and a two-sided t-test follow the codebook convention.
    fit = smf.ols("earnings_m6 ~ treatment", data=site_data).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
