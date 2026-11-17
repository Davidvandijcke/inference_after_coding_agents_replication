#!/usr/bin/env python3
"""Run the fixed Pathways to Work 12-month earnings analysis.

Requires pandas, numpy, scipy, and statsmodels. The treatment coefficient is
estimated by OLS with site indicators, baseline earnings, and baseline employment.
Inference uses HC2 standard errors and a two-sided residual-df t-test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    data = pd.read_csv(data_path)
    columns = [
        "earnings_m12", "treatment", "site", "earnings_baseline",
        "employed_baseline",
    ]
    data = data.loc[:, columns]
    if data.empty or data.isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Analysis variables must be finite numeric values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both randomized arms, coded 0/1.")

    model = smf.ols(
        "earnings_m12 ~ treatment + C(site) + earnings_baseline + employed_baseline",
        data=data,
        missing="raise",
    )
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        raise ValueError("The specified regression design must have full rank.")
    result = model.fit(cov_type="HC2", use_t=True)
    if result.df_resid <= 0:
        raise ValueError("The specified regression needs positive residual degrees of freedom.")
    return {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    values = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
