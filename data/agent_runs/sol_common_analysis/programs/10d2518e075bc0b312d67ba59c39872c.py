#!/usr/bin/env python3
"""Run the fixed 12-month debt-collections intention-to-treat analysis.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels (which depends on scipy).

Model: debt_collections_m12 ~ intercept + treatment + site indicators.
Inference: HC2 covariance, two-sided Student t test, residual degrees of
freedom n - rank(X), as specified by CODEBOOK.md. The JSON p_value is the
UNADJUSTED p-value for this one fixed test. This outcome was selected from
18 outcomes in the original data; the original finding's Bonferroni-adjusted
p-value is min(1, 18 * p_value). See analysis/report.md for interpretation.
No outcome selection or other analyses are performed by this program.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    """Estimate the fixed site-adjusted effect of randomized assignment."""
    columns = ["site", "treatment", "debt_collections_m12"]
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis columns must be nonempty and complete.")
    for column in ["treatment", "debt_collections_m12"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both randomized assignment groups are required.")

    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["treatment"]].astype(float),
            pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float),
        ],
        axis=1,
    )
    if (len(design) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]):
        raise ValueError("The site-adjusted model requires full rank and positive residual degrees of freedom.")

    fitted = sm.OLS(data["debt_collections_m12"], design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The model did not produce finite estimates and inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
