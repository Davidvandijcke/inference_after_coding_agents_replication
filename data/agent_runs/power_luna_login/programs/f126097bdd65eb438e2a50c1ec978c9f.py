#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    site_indicators = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    x = pd.concat(
        [data[["treatment"]].astype(float), site_indicators], axis=1
    )
    x = sm.add_constant(x)
    model = sm.OLS(data["employed_m12"].astype(float), x).fit(
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
