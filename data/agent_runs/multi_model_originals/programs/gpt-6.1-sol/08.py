#!/usr/bin/env python3
"""Fixed analysis: assignment effect on training_m12 among parent == 1.

Model: training_m12 ~ intercept + treatment + categorical site indicators.
Inference: HC2 standard error and two-sided Student t test, with n - rank(X)
degrees of freedom. Estimates and standard errors are in probability units.
This program performs no outcome, subgroup, or specification search.

Requires pandas, numpy, and statsmodels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ["parent", "treatment", "site", "training_m12"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if data[required].isna().any().any():
        raise ValueError("Required columns must have no missing observations.")
    for column in ("parent", "treatment", "training_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be coded 0 or 1.")

    parents = data.loc[data["parent"].eq(1)]
    if parents.empty:
        raise ValueError("No participants have parent == 1.")
    sites = pd.get_dummies(parents["site"], prefix="site", drop_first=True,
                           dtype=float)
    X = sm.add_constant(pd.concat([
        parents[["treatment"]].astype(float), sites
    ], axis=1), has_constant="add")
    if np.linalg.matrix_rank(X.to_numpy()) != X.shape[1]:
        raise ValueError("Treatment effect is not identified within sites.")
    if len(parents) <= X.shape[1]:
        raise ValueError("No residual degrees of freedom.")

    fit = sm.OLS(parents["training_m12"].astype(float), X).fit(
        cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The HC2 standard error or t test is undefined.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must differ from the input data path.")
    try:
        result = score(args.data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
