#!/usr/bin/env python3
"""Score the selected lead analysis for a Pathways to Work dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_m12", "treatment", "site", "employed_baseline"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    # Site-adjusted OLS with the baseline employment measure; HC2 standard
    # errors and a two-sided t-test follow the convention in CODEBOOK.md.
    result = smf.ols(
        "employed_m12 ~ treatment + C(site) + employed_baseline", data=data
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
