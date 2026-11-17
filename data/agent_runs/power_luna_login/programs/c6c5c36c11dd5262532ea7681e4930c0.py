#!/usr/bin/env python3
"""Score the 12-month employment treatment effect using the study convention."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file with evaluation observations")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = {"employed_m12", "treatment", "site"}
    missing = needed.difference(data.columns)
    if missing:
        parser.error("data is missing required columns: " + ", ".join(sorted(missing)))
    if data[list(needed)].isna().any().any():
        parser.error("required analysis columns must not contain missing values")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(cov_type="HC2")
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    p_value = float(2 * t.sf(abs(estimate / std_error), model.df_resid))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
