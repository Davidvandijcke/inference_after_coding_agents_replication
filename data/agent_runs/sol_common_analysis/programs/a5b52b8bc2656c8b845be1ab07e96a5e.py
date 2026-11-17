#!/usr/bin/env python3
"""Estimate the site-adjusted intention-to-treat effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels.
Inference: OLS, HC2 covariance, two-sided Student t test with OLS residual df.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        parser.error("The analysis requires nonempty, complete observations.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            parser.error(f"{column} must contain only 0 and 1.")
    if data.treatment.nunique() != 2:
        parser.error("Both treatment and control observations are required.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data, missing="raise")
    if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
        parser.error("The site-adjusted treatment effect is not identifiable.")
    if len(data) <= model.exog.shape[1]:
        parser.error("Positive residual degrees of freedom are required.")
    fit = model.fit(cov_type="HC2", use_t=True)
    if np.any(fit.get_influence().hat_matrix_diag >= 1 - 1e-12):
        parser.error("HC2 inference requires all observation leverages below one.")
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
