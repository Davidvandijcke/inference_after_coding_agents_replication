#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

Estimates the treatment effect on 12-month food insecurity among participants
coded as not female, controlling for randomization site. Uses an OLS linear
probability model and HC2 robust standard errors, as specified in CODEBOOK.md.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["female"] == 0]
    model = smf.ols(
        "food_insecure_m12 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

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
