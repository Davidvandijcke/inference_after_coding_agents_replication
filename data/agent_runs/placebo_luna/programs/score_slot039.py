#!/usr/bin/env python3
"""Score the year-one energy outcome in participants aged 50 or older.

Fits OLS of lots_of_energy_y1 on treatment and randomization-block indicators
among age_group == 3, using HC2 standard errors and a two-sided t test.
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
    required = {"age_group", "treatment", "block", "lots_of_energy_y1"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("data is missing required columns: " + ", ".join(sorted(missing)))

    analysis = data.loc[data["age_group"] == 3,
                        ["lots_of_energy_y1", "treatment", "block"]].dropna()
    if analysis.empty or analysis["treatment"].nunique() != 2:
        parser.error("the age_group == 3 analysis requires observed outcomes in both arms")

    fit = smf.ols("lots_of_energy_y1 ~ treatment + C(block)", data=analysis).fit(cov_type="HC2")
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
