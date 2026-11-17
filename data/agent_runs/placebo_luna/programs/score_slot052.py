#!/usr/bin/env python3
"""Estimate the block-adjusted treatment effect on follow-up relationship satisfaction."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "satisfied_relationship_fu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file with the trial data")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"treatment", "block", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    analysis = data[["treatment", "block", OUTCOME]].dropna()
    if analysis["treatment"].nunique() != 2:
        raise ValueError("The observed outcome data must contain both treatment arms")

    model = smf.ols(f'{OUTCOME} ~ treatment + C(block)', data=analysis).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
