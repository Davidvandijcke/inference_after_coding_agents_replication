#!/usr/bin/env python3
"""Run the frozen Pathways to Work analysis, without a search.

Estimand: the site-adjusted intention-to-treat effect of program assignment on
debt_collections_m12 among participants with employed_baseline == 1.

Model: OLS with an intercept, treatment, and indicators for observed sites.
Inference: HC2 standard error and a two-sided Student t test using n - rank(X)
residual degrees of freedom. Estimates are in probability units (0.01 = 1 pp).

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data):
    required = ["employed_baseline", "debt_collections_m12", "treatment", "site"]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        raise ValueError("The chosen analysis requires complete data in its four columns.")
    if not data["employed_baseline"].isin([0, 1]).all():
        raise ValueError("employed_baseline must be coded 0 or 1.")

    sample = data.loc[data["employed_baseline"].eq(1)]
    if set(sample["treatment"].unique()) != {0, 1}:
        raise ValueError("The baseline-employed subgroup must contain both treatment arms.")
    if not sample["debt_collections_m12"].isin([0, 1]).all():
        raise ValueError("debt_collections_m12 must be coded 0 or 1.")

    sites = pd.get_dummies(
        sample["site"].astype("category"),
        prefix="site", drop_first=True, dtype=float,
    )
    design = pd.concat([sample[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    matrix = design.to_numpy(dtype=float)
    if len(sample) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("The subgroup regression needs full rank and positive residual degrees of freedom.")
    q, _ = np.linalg.qr(matrix, mode="reduced")
    if np.any(np.sum(q * q, axis=1) >= 1 - 1e-12):
        raise ValueError("HC2 is undefined when an observation has leverage one.")

    fit = sm.OLS(sample["debt_collections_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True,
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The selected regression did not yield finite estimates and inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ.")
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
