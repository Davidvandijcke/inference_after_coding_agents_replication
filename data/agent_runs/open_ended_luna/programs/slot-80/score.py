#!/usr/bin/env python3
"""Score the chosen exploratory finding in a Pathways to Work dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the site-adjusted treatment effect on 12-month food "
            "insecurity among participants who were parents at baseline."
        )
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"parent", "treatment", "site", "food_insecure_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error(f"input CSV is missing required columns: {', '.join(missing)}")

    parents = data.loc[data["parent"] == 1]
    if parents.empty:
        parser.error("input CSV has no participants with parent == 1")

    result = smf.ols(
        "food_insecure_m12 ~ treatment + C(site)", data=parents
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
