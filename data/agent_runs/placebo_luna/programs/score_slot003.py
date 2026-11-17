#!/usr/bin/env python3
"""Score the year-2 effect on working 50 or more hours per week."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "works_50_plus_hours_y2"
BASELINE = "works_50_plus_hours_base"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {OUTCOME, "treatment", "block", BASELINE}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input is missing required columns: " + ", ".join(sorted(missing)))

    analysis = data[[OUTCOME, "treatment", "block", BASELINE]].dropna()
    if analysis.empty:
        parser.error("no complete observations for the selected analysis")

    model = smf.ols(
        f"{OUTCOME} ~ treatment + C(block) + {BASELINE}", data=analysis
    ).fit(cov_type="HC2")
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
