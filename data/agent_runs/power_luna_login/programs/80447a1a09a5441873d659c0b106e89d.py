#!/usr/bin/env python3
"""Estimate the 12-month employment effect of Pathways to Work.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    # Site fixed effects reflect the within-site randomization. HC2 standard
    # errors and use_t=True implement the codebook's robust two-sided t-test.
    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
