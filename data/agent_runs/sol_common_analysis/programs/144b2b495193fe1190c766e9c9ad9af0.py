#!/usr/bin/env python3
"""Estimate the six-month training intention-to-treat effect.

Requires pandas and statsmodels. The reported p-value is the unadjusted,
two-sided HC2 t-test p-value for treatment in an OLS regression with an
intercept and site indicators. No other outcomes or models are analyzed.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=["site", "treatment", "training_m6"])
    if data.empty or data.isna().any().any():
        raise ValueError("The selected analysis requires complete observations.")
    if not data["treatment"].isin([0, 1]).all() or data["treatment"].nunique() != 2:
        raise ValueError("Treatment must include both 0 and 1.")
    if not data["training_m6"].isin([0, 1]).all():
        raise ValueError("training_m6 must be binary.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    ).astype(float)
    result = sm.OLS(data["training_m6"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
