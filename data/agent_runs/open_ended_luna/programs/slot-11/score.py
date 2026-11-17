#!/usr/bin/env python3
"""Score the prespecified distress-by-baseline-employment interaction."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate treatment-effect heterogeneity in 6-month distress "
            "by baseline employment, adjusting for site."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_baseline", "distress_m6"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error(f"data is missing required columns: {', '.join(missing)}")
    if data[list(required)].isna().any().any():
        parser.error("analysis columns must not contain missing values")

    data = data.copy()
    data["baseline_nonemployment"] = 1 - data["employed_baseline"]
    fit = smf.ols(
        "distress_m6 ~ treatment * baseline_nonemployment + C(site)",
        data=data,
    ).fit(cov_type="HC2", use_t=True)

    interaction = "treatment:baseline_nonemployment"
    result = {
        "estimate": float(fit.params[interaction]),
        "std_error": float(fit.bse[interaction]),
        "p_value": float(fit.pvalues[interaction]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
