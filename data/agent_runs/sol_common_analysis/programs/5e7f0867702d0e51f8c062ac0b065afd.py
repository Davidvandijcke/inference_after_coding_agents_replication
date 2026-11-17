#!/usr/bin/env python3
"""Estimate the site-adjusted intention-to-treat effect on 12-month employment.

Requires pandas and statsmodels. Output is in probability units (0.01 = 1 pp).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty data without missing values.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data,
                    missing="raise")
    if (np.linalg.matrix_rank(model.exog) < model.exog.shape[1]
            or model.exog.shape[0] <= model.exog.shape[1]):
        raise ValueError("The regression requires full rank and residual degrees of freedom.")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(output, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
