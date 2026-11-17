#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on market-offset explanations.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"block", "treatment", "reason_market_offsets"}
    missing = sorted(required - set(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    # OLS with randomization-block indicators and HC2 heteroskedasticity-robust SE.
    model = smf.ols(
        "reason_market_offsets ~ treatment + C(block)", data=data
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
