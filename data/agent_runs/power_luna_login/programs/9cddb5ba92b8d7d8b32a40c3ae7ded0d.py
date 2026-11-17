#!/usr/bin/env python3
"""Score the prespecified Pathways to Work 12-month employment analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(sorted(missing)))

    # OLS with site indicators; HC2 heteroskedasticity-robust inference.
    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(cov_type="HC2")
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
