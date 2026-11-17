#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on follow-up family-time satisfaction."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file with evaluation data")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    outcome = "satisfied_family_time_fu"
    required = [outcome, "treatment", "block"]
    data = pd.read_csv(args.data)
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    analysis = data[required].dropna()
    if analysis.empty:
        parser.error("no complete observations for the selected analysis")

    result = smf.ols(
        'Q("satisfied_family_time_fu") ~ treatment + C(block)', data=analysis
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
