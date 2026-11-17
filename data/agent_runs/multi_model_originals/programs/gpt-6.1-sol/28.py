#!/usr/bin/env python3
"""Fixed exploratory finding, for replication on data with the packet's schema.

Among participants with education 3 or 4, regress debt_collections_m6 on
treatment and categorical site indicators, with an intercept and no other
covariates. Return the treatment coefficient (probability units), HC2 standard
error, and two-sided Student t p-value using n - rank(X) degrees of freedom.
No outcome, subgroup, follow-up, or specification search is performed.

Dependencies: numpy, pandas, statsmodels (and its scipy dependency).
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data):
    columns = ["education", "site", "treatment", "debt_collections_m6"]
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[columns].isna().any().any():
        raise ValueError("Required columns must have no missing values.")
    if not data.education.isin([1, 2, 3, 4]).all():
        raise ValueError("education must use codes 1, 2, 3, 4.")
    d = data.loc[data.education.isin([3, 4])].copy()
    if d.empty:
        raise ValueError("No participants have education 3 or 4.")
    if not d.treatment.isin([0, 1]).all() or d.treatment.nunique() != 2:
        raise ValueError("The chosen subgroup must contain treatment 0 and 1.")
    if not d.debt_collections_m6.isin([0, 1]).all():
        raise ValueError("debt_collections_m6 must be binary (0 or 1).")
    x = pd.concat([
        pd.Series(1., index=d.index, name="const"),
        d[["treatment"]].astype(float),
        pd.get_dummies(d.site, prefix="site", drop_first=True, dtype=float),
    ], axis=1)
    matrix = x.to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(matrix)
    if rank != matrix.shape[1] or len(d) <= rank:
        raise ValueError("The subgroup design must have full rank and positive residual degrees of freedom.")
    leverage = np.sum(np.linalg.qr(matrix, mode="reduced")[0] ** 2, axis=1)
    if np.any(1. - leverage <= np.finfo(float).eps * 10):
        raise ValueError("HC2 inference is undefined for an observation with unit leverage.")
    model = sm.OLS(d.debt_collections_m6.astype(float), x).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()) or result["std_error"] <= 0:
        raise ValueError("The chosen regression does not provide finite, nondegenerate inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    try:
        result = chosen_analysis(pd.read_csv(args.data))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
