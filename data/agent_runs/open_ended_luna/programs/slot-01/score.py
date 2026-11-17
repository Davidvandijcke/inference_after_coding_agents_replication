#!/usr/bin/env python3
"""Score the selected 12-month earnings heterogeneity analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input participant CSV")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"earnings_m12", "treatment", "employed_baseline", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error(f"input is missing required columns: {', '.join(sorted(missing))}")

    # The site-by-baseline-employment terms allow site intercepts to differ
    # across the two baseline groups. The treatment interaction tests the
    # difference in their 12-month earnings effects. Orient the estimate as
    # effect among baseline-unemployed minus effect among baseline-employed.
    model = smf.ols(
        "earnings_m12 ~ treatment * employed_baseline "
        "+ C(site) * employed_baseline",
        data=data,
    ).fit(cov_type="HC2", use_t=True)
    interaction = "treatment:employed_baseline"
    result = {
        "estimate": float(-model.params[interaction]),
        "std_error": float(model.bse[interaction]),
        "p_value": float(model.pvalues[interaction]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
