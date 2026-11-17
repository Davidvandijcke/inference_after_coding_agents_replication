#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment.

Requires pandas and statsmodels. Uses OLS with site fixed effects, HC2
standard errors, and a two-sided Student t test with residual degrees of
freedom, as specified in CODEBOOK.md. Estimates are in probability units.
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
    required = ["employed_m12", "treatment", "site"]
    missing = set(required) - set(data.columns)
    if missing:
        parser.error(f"Missing columns: {', '.join(sorted(missing))}")
    if data[required].isna().any().any():
        parser.error("Analysis columns must be fully observed")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must contain only 0 and 1")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        parser.error("Treatment effect is not identified with site indicators")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        parser.error("Dataset does not support finite estimates and inference")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
