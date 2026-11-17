#!/usr/bin/env python3
"""Run the selected site-adjusted employment analysis among women."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"female", "employed_m12", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    # Chosen analysis: among women, regress 12-month employment on treatment
    # and site indicators, using the codebook's HC2 standard errors and a
    # two-sided t-test for the treatment coefficient.
    subgroup = data.loc[data["female"] == 1]
    if subgroup.empty or subgroup["treatment"].nunique() != 2:
        parser.error("female subgroup must contain observations in both treatment arms")

    result = smf.ols("employed_m12 ~ treatment + C(site)", data=subgroup).fit(
        cov_type="HC2", use_t=True
    )
    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
