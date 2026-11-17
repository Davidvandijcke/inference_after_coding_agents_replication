#!/usr/bin/env python3
"""Score the 12-month employment effect in a Pathways to Work dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("missing required columns: " + ", ".join(sorted(missing)))

    result = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
