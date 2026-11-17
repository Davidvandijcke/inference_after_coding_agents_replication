#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output results.json
Requires pandas, numpy, and statsmodels (which depends on scipy).

The only fitted model is employed_m12 ~ treatment + C(site), with an
intercept, HC2 standard errors, and a two-sided Student t test using
residual degrees of freedom. The reported p_value is nominal, without
adjustment for the exploratory outcome screening in the original analysis.
The estimate and standard error are on the probability (0 to 1) scale.
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

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        parser.error("The analysis columns must contain complete observations")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1, and no other values")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must contain only 0 and 1")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    design = model.exog
    if design.shape[0] <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        parser.error("The model requires a full-rank design and positive residual degrees of freedom")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        parser.error("The chosen model did not produce finite inference")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
