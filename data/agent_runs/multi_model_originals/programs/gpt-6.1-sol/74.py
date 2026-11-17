#!/usr/bin/env python3
"""Run the fixed, exploratory Pathways to Work analysis.

Estimand: assignment effect on employed_m12 among participants with parent == 0.
Model: OLS with intercept, treatment, and categorical site indicators; HC2
standard error; two-sided Student t test using residual degrees of freedom.
The reported p-value is nominal, without adjustment for the prior exploration.

Dependencies: numpy, pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    columns = ["parent", "treatment", "site", "employed_m12"]
    data = pd.read_csv(data_path, usecols=columns)
    if data[columns].isna().any().any():
        raise ValueError("The four analysis columns must be fully observed.")
    for col in ("parent", "treatment", "employed_m12"):
        if not data[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 and 1.")
    sample = data.loc[data["parent"].eq(0)].copy()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("The selected subgroup must contain both trial arms.")
    sites = pd.get_dummies(sample["site"], prefix="site",
                           drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([sample[["treatment"]], sites], axis=1),
        has_constant="add",
    ).astype(float)
    if len(sample) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The subgroup model needs full rank and positive residual degrees of freedom.")
    fit = sm.OLS(sample["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The selected model produced nonfinite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
