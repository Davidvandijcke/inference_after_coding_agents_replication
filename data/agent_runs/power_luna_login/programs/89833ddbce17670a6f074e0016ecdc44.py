#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment treatment effect.

Usage: python score.py --data <path> --output <path>

Fits OLS of employed_m12 on treatment and site indicators, then reports the
HC2-robust treatment estimate, standard error, and two-sided t-test p-value.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_m12", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    result = smf.ols('Q("employed_m12") ~ treatment + C(site)', data=data).fit(
        cov_type="HC2", use_t=True
    )
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
