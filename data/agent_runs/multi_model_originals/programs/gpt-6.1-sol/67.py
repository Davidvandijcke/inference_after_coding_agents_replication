#!/usr/bin/env python3
"""Run the fixed, exploratory Pathways to Work analysis.

Test: assignment effect on debt_collections_m12 among participants whose
baseline education is 1 or 2, adjusting for categorical site indicators.
OLS, HC2 standard error, two-sided Student t test with residual model df.
The estimate and standard error are probability units, not percentage points.

Requires Python 3, numpy, pandas and statsmodels (which requires scipy).
Usage: python score.py --data data.csv --output analysis/chosen_result.json
This program does not search outcomes, subgroups or model specifications.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_test(data_path):
    columns = ["site", "treatment", "education", "debt_collections_m12"]
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError("Required columns must be nonempty and fully observed.")
    if not data["education"].isin([1, 2, 3, 4]).all():
        raise ValueError("education must use the codebook values 1 through 4.")
    for column in ["treatment", "debt_collections_m12"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    sample = data.loc[data["education"].isin([1, 2])].copy()
    if sample["treatment"].nunique() != 2:
        raise ValueError("The chosen subgroup must contain both treatment arms.")
    sites = pd.get_dummies(sample["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = sm.add_constant(pd.concat([sample[["treatment"]], sites], axis=1),
                             has_constant="add").astype(float)
    if (len(sample) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]):
        raise ValueError("The chosen model needs positive residual df and full rank.")
    fit = sm.OLS(sample["debt_collections_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The chosen test did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must be separate from the input dataset.")
    result = chosen_test(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
