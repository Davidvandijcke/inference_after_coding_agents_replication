#!/usr/bin/env python3
"""Run the fixed exploratory finding selected in the Pathways to Work analysis.

Analysis: food_insecure_m6 ~ treatment + C(site), restricted to female == 0.
The coefficient is a probability difference, not percentage points.
Inference: OLS, HC2 covariance, two-sided Student t with residual degrees of
freedom. The reported p-value is nominal; it does not adjust for selection.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels (and their scipy dependency).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    required = ["female", "site", "treatment", "food_insecure_m6"]
    data = pd.read_csv(data_path, usecols=required)
    if data[required].isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    for column in ["female", "treatment", "food_insecure_m6"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be coded 0 or 1.")
    subgroup = data.loc[data["female"].eq(0)].copy()
    if subgroup.empty or subgroup["treatment"].nunique() != 2:
        raise ValueError("The female=0 subgroup must contain both treatment arms.")
    model = smf.ols("food_insecure_m6 ~ treatment + C(site)", data=subgroup)
    design = model.exog
    if len(subgroup) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The selected regression needs a full-rank design and residual degrees of freedom.")
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError("HC2 inference is undefined for an observation with leverage one.")
    fitted = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Regression returned non-finite results.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
