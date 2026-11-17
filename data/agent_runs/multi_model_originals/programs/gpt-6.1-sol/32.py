#!/usr/bin/env python3
"""Fixed analysis: 12-month employment ITT, site indicators, HC2, two-sided t.

Dependencies: numpy, pandas, scipy, statsmodels.
Example: python score.py --data data.csv --output result.json
Estimates and standard errors are in probability units (0.01 = 1 percentage point).
No subgroup, outcome, time-point, or covariate selection is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ["employed_m12", "treatment", "site"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[required].copy()
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must be nonempty and fully observed.")
    for column in ["employed_m12", "treatment"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both randomized assignment groups are required.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment must be estimable after adjusting for site.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("This dataset does not yield finite HC2 inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
