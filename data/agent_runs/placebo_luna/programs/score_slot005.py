#!/usr/bin/env python3
"""Score the CO2-topic subgroup treatment effect on market-offset reasoning."""

import argparse
import json
import sys

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.data)
        required = {"wtp_topic", "reason_market_offsets", "treatment", "block"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError("missing required columns: " + ", ".join(sorted(missing)))

        subset = data.loc[data["wtp_topic"].eq("co2"),
                          ["reason_market_offsets", "treatment", "block"]].dropna()
        if subset["treatment"].nunique() != 2:
            raise ValueError("CO2 subgroup must contain both treatment arms")

        result = smf.ols(
            "reason_market_offsets ~ treatment + C(block)", data=subset
        ).fit(cov_type="HC2", use_t=True)
        payload = {
            "estimate": float(result.params["treatment"]),
            "std_error": float(result.bse["treatment"]),
            "p_value": float(result.pvalues["treatment"]),
        }
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
    except Exception as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
