#!/usr/bin/env python3
"""Run the fixed, exploratory Pathways to Work finding on a new CSV.

Selected analysis: participants with female == 0; outcome food_insecure_m12;
OLS on an intercept, treatment, and site indicators, without other covariates.
The treatment coefficient is a probability difference (not percentage points).
Inference: HC2 standard error and a two-sided Student t test with residual df.
No outcomes, follow-ups, subgroups, or model specifications are searched.

Requires numpy, pandas, scipy, and statsmodels.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    columns = ["female", "site", "treatment", "food_insecure_m12"]
    data = pd.read_csv(data_path, usecols=columns)
    if data[columns].isna().any().any():
        raise ValueError("Required analysis columns must have no missing values.")
    for column in ["female", "treatment", "food_insecure_m12"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    sample = data.loc[data["female"].eq(0)].copy()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("The female=0 subgroup must include both treatment arms.")
    design = pd.concat([
        pd.Series(1.0, index=sample.index, name="const"),
        sample[["treatment"]].astype(float),
        pd.get_dummies(sample["site"].astype(str), prefix="site",
                       drop_first=True, dtype=float),
    ], axis=1)
    if (len(sample) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]):
        raise ValueError("Insufficient within-site treatment variation or residual df.")
    fit = sm.OLS(sample["food_insecure_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The selected regression produced undefined inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data; inputs are never overwritten.")
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
