#!/usr/bin/env python3
"""Estimate the site-adjusted intention-to-treat effect on 12-month employment.

Dependencies: pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), sites], axis=1),
        has_constant="add",
    )
    model = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
