#!/usr/bin/env python3
"""Fixed Pathways to Work analysis: 12-month employment, all participants.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels (and its scipy dependency).
The reported treatment effect and standard error are in probability units.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    """OLS of employed_m12 on assignment and site indicators; HC2, t test."""
    data = pd.read_csv(data_path)
    required = ["employed_m12", "treatment", "site"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data.loc[:, required].copy()
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis requires nonempty data with no missing analysis values.")
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Analysis variables must be finite.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Both treatment arms, coded 0 and 1, must be present.")
    if not data["site"].isin([1, 2, 3, 4]).all():
        raise ValueError("site must be coded 1, 2, 3, or 4.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data, missing="raise")
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment and site coefficients are not separately identifiable.")
    if len(data) <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    # HC2 divides each squared residual by (1 - its leverage).
    q, _ = np.linalg.qr(design, mode="reduced")
    if np.any(1.0 - np.sum(q * q, axis=1) <= 1e-12):
        raise ValueError("HC2 is undefined for observations with unit leverage.")

    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The selected analysis did not yield finite results.")
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
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
