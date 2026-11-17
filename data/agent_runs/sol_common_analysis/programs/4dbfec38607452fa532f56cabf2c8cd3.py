#!/usr/bin/env python3
"""Estimate the fixed, full-sample effect on 12-month self-rated health.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, scipy, and statsmodels.
Model: health_m12 ~ treatment + C(site), with an intercept, HC2 standard
errors, and a two-sided t test using residual degrees of freedom.
The p_value is nominal: this program performs no outcome selection or
multiple-testing adjustment.
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
        parser.error("The output path must differ from the input dataset.")

    data = pd.read_csv(args.data, usecols=["health_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        parser.error("The analysis requires nonempty, complete observations.")
    for column in ("health_m12", "treatment"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy()).all():
            parser.error(f"{column} must contain only finite values.")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1, and no other values.")

    model = smf.ols("health_m12 ~ treatment + C(site)", data=data,
                    missing="raise")
    design = model.exog
    if (design.shape[0] <= design.shape[1]
            or np.linalg.matrix_rank(design) != design.shape[1]):
        parser.error("The model requires a full-rank design and positive residual df.")
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        parser.error("HC2 is undefined for observations with leverage one.")

    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        parser.error("The analysis produced undefined inference.")
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
