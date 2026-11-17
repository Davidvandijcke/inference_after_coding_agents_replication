#!/usr/bin/env python3
"""Score the selected Pathways to Work exploratory finding.

Estimates the treatment effect on six-month food insecurity among participants
with some college education (education == 3), adjusting for site indicators.
Inference uses HC2 robust standard errors and a two-sided t test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"education", "food_insecure_m6", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    subgroup = data.loc[data["education"] == 3]
    if subgroup.empty:
        parser.error("no participants with education == 3 in the input data")

    # The contrast follows the codebook: OLS of the outcome on treatment and
    # site indicators, with HC2 standard errors and a two-sided t test.
    result = smf.ols(
        "food_insecure_m6 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
