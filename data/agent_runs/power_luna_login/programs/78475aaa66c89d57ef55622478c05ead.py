#!/usr/bin/env python3
"""Score the prespecified exploratory finding from the Pathways evaluation.

Fits employed_m12 on treatment and site indicators among participants who
were not employed at baseline, using HC2 robust standard errors and a
two-sided t-test for the treatment coefficient.
"""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subset = data.loc[data["employed_baseline"] == 0]

    site_indicators = pd.get_dummies(
        subset["site"], prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=subset.index, name="const"),
            subset["treatment"].astype(float).rename("treatment"),
            site_indicators,
        ],
        axis=1,
    )
    model = sm.OLS(subset["employed_m12"].astype(float), design).fit(
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
