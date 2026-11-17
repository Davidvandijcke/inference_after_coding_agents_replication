#!/usr/bin/env python3
"""Run the prespecified lead analysis on a dataset with the evaluation columns."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "if_partner_flexible_post"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {OUTCOME, "treatment", "block"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(sorted(missing)))

    analysis = data[[OUTCOME, "treatment", "block"]].dropna()
    if analysis["treatment"].nunique() != 2:
        parser.error("analysis data must contain both treatment groups")

    model = smf.ols(f"{OUTCOME} ~ treatment + C(block)", data=analysis).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
