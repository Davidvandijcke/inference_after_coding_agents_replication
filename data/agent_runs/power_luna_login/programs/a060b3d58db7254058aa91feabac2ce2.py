#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

Estimates the treatment effect on usual weekly hours at month 12 among
participants with less than a high-school education, adjusting for site.
"""

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
    subgroup = data.loc[data["education"] == 1]
    if subgroup.empty or subgroup["treatment"].nunique() != 2:
        raise ValueError("The education == 1 subgroup must contain both treatment arms")

    model = smf.ols("hours_m12 ~ treatment + C(site)", data=subgroup).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
