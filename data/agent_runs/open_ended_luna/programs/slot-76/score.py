#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis on a CSV dataset.

Estimates the treatment effect on 12-month food insecurity among participants
with a bachelor's degree or higher, using OLS with site indicators and HC2
heteroskedasticity-robust standard errors.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


OUTCOME = "food_insecure_m12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input participant-level CSV")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"education", "treatment", "site", OUTCOME}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    subgroup = data.loc[data["education"].eq(4)]
    if subgroup.empty:
        parser.error("input CSV has no participants with education == 4")

    result = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=subgroup).fit(
        cov_type="HC2", use_t=True
    )
    estimate = float(result.params["treatment"])
    std_error = float(result.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=float(result.df_resid)))

    output = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
