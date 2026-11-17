#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on 12-month employment only.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels. Estimates are in probability units.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")

    data = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        parser.error("The analysis columns must contain complete observations")
    for column in ["employed_m12", "treatment"]:
        if not data[column].isin([0, 1]).all():
            parser.error(f"{column} must contain only 0 and 1")
    if data["treatment"].nunique() != 2:
        parser.error("Both treatment groups must be present")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    design = model.exog
    if design.shape[0] <= design.shape[1]:
        parser.error("Positive residual degrees of freedom are required")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        parser.error("Treatment and site indicators must be jointly identifiable")

    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        parser.error("The HC2 estimate, standard error, and p-value must be finite")
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
