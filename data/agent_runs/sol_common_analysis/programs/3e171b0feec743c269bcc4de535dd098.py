#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment, adjusting for site.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, scipy, statsmodels.
Estimates and standard errors are in probability units, not percentage points.
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

    data = pd.read_csv(args.data)
    columns = ["employed_m12", "treatment", "site"]
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        parser.error(f"Missing required columns: {', '.join(missing)}")
    if data[columns].isna().any().any():
        parser.error("Analysis columns must be fully observed")
    if not data["treatment"].isin([0, 1]).all() or data["treatment"].nunique() != 2:
        parser.error("treatment must contain both 0 and 1, and no other values")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must contain only 0 and 1")

    model = smf.ols(
        "employed_m12 ~ treatment + C(site)", data=data, missing="raise"
    )
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        parser.error("The treatment and site regression is not identifiable")
    if len(data) <= design.shape[1]:
        parser.error("Insufficient observations for residual degrees of freedom")

    # use_t=True explicitly requests the codebook's two-sided t-test.
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        parser.error("The regression did not produce finite inference")
    args.output.write_text(json.dumps(output, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
