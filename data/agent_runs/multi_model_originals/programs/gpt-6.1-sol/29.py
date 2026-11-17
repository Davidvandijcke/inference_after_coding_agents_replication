#!/usr/bin/env python3
"""Frozen Pathways to Work analysis: 12-month psychological distress.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels (and its scipy dependency).

All participants; OLS with an intercept, treatment, and site indicators.
HC2 standard errors and a two-sided Student t test, using residual degrees
of freedom. The p-value is nominal: it does not adjust for the exploration
that led to choosing this outcome. Negative estimates favor the program.
No subgroup selection, covariate selection, or outcome search is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "distress_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The selected analysis requires nonempty, complete data.")
    treatment = pd.to_numeric(data["treatment"], errors="raise").astype(float)
    outcome = pd.to_numeric(data["distress_m12"], errors="raise").astype(float)
    if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
        raise ValueError("Treatment and distress_m12 must be finite numbers.")
    if set(treatment.unique()) != {0.0, 1.0}:
        raise ValueError("Treatment must contain both 0 and 1, with no other values.")

    site_indicators = pd.get_dummies(
        data["site"].astype("category"), prefix="site", drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [pd.Series(1.0, index=data.index, name="const"),
         treatment.rename("treatment"), site_indicators], axis=1,
    )
    if len(data) <= design.shape[1]:
        raise ValueError("There are insufficient residual degrees of freedom.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment cannot be identified separately from site.")

    model = sm.OLS(outcome, design)
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The HC2 estimate or inference is undefined for these data.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    try:
        result = analyze(args.data)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
