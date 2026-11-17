#!/usr/bin/env python3
"""Estimate the full-sample intention-to-treat effect on employed_m12.

Dependencies: pandas, statsmodels (and their dependencies).
Usage: python score.py --data data.csv --output result.json
The estimate and standard error are in probability units.
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
        parser.error("--output must differ from --data")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if not data.groupby("site")["treatment"].nunique().eq(2).all():
        raise ValueError("Each site must contain both randomized groups.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    if len(data) <= model.exog.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")
    result = model.fit(cov_type="HC2", use_t=True)
    values = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("The chosen model did not yield finite inference.")
    args.output.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
