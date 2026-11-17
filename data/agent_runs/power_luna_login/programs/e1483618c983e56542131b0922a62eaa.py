#!/usr/bin/env python3
"""Score the prespecified follow-up analysis selected from the supplied packet.

Estimates the site-adjusted treatment effect on 6-month food insecurity among
participants unemployed at baseline, using HC2 standard errors.
"""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["employed_baseline"] == 0]

    site_indicators = pd.get_dummies(
        subgroup["site"], prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat(
        [subgroup[["treatment"]].astype(float), site_indicators], axis=1
    )
    design = sm.add_constant(design)
    result = sm.OLS(subgroup["food_insecure_m6"], design).fit(
        cov_type="HC2", use_t=True
    )

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
