#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis; no outcome or subgroup search.

Population: employed_baseline == 0.
Outcome: employed_m6 (0/1, so estimates are probability differences).
Model: OLS on an intercept, treatment, and categorical site indicators.
Inference: HC2 standard error; two-sided Student t test with n - rank(X) df.

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data_path):
    columns = ["site", "treatment", "employed_baseline", "employed_m6"]
    data = pd.read_csv(data_path, usecols=columns)
    if data.isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    for column in ["treatment", "employed_baseline", "employed_m6"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    subgroup = data.loc[data["employed_baseline"] == 0].copy()
    if subgroup["treatment"].nunique() != 2:
        raise ValueError("The chosen subgroup must include both treatment arms.")

    model = smf.ols("employed_m6 ~ treatment + C(site)", data=subgroup)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment and site effects are not separately identifiable.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("The chosen model needs positive residual degrees of freedom.")
    q, _ = np.linalg.qr(design, mode="reduced")
    if np.any(1.0 - np.sum(q * q, axis=1) <= 1e-12):
        raise ValueError("HC2 inference is undefined for an observation with leverage 1.")

    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The chosen model did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    try:
        result = chosen_analysis(args.data)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
