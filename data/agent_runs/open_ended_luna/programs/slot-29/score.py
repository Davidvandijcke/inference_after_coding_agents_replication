#!/usr/bin/env python3
"""Estimate the fixed 12-month treatment effect on psychological distress."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted ITT effect on distress_m12."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    if data_path.resolve() == output_path.resolve():
        parser.error("--output must not overwrite --data")

    data = pd.read_csv(data_path)
    required = {"distress_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input is missing required columns: " + ", ".join(sorted(missing)))

    result = smf.ols("distress_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
