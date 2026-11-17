#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

Estimates the treatment effect on 12-month employment among participants who
were not employed at baseline, adjusting for randomization site.
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
    required = {"employed_baseline", "employed_m12", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    subgroup = data.loc[data["employed_baseline"] == 0]
    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=subgroup).fit(
        cov_type="HC2", use_t=True
    )
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
