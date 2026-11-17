#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the site-adjusted treatment effect on six-month training "
            "among participants with less than a high-school education."
        )
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    if data_path == output_path:
        parser.error("--output must be a different file from --data")

    data = pd.read_csv(data_path)
    required = {"education", "training_m6", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error(f"Input is missing required columns: {', '.join(sorted(missing))}")

    subgroup = data.loc[data["education"] == 1]
    fit = smf.ols("training_m6 ~ treatment + C(site)", data=subgroup).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
