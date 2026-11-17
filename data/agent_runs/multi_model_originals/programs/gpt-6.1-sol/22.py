#!/usr/bin/env python3
"""Rerun one fixed exploratory Pathways to Work analysis.

Estimand: the intention-to-treat coefficient for fin_security_m12 among
participants with female == 0, controlling for site with an intercept and
site indicators. Inference: OLS, HC2 standard error, two-sided Student t test
with n - rank(X) residual degrees of freedom. The reported p-value is nominal;
it does not account for the exploratory selection of this analysis.

Dependencies: numpy, pandas, statsmodels (and their dependencies).
Usage: python score.py --data data.csv --output chosen_result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    """Fit only the fixed outcome, subgroup, and specification above."""
    columns = ["female", "site", "treatment", "fin_security_m12"]
    data = pd.read_csv(data_path, usecols=columns)
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError("Required analysis columns must be observed and finite.")
    if not data["female"].isin([0, 1]).all():
        raise ValueError("female must be coded 0 or 1.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must be coded 0 or 1.")
    if not data["site"].isin([1, 2, 3, 4]).all():
        raise ValueError("site must be coded 1, 2, 3, or 4.")

    subgroup = data.loc[data["female"].eq(0)].copy()
    if subgroup["treatment"].nunique() != 2:
        raise ValueError("The female == 0 subgroup must contain both trial arms.")
    site_indicators = pd.get_dummies(
        subgroup["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([subgroup[["treatment"]].astype(float), site_indicators], axis=1),
        has_constant="add",
    )
    matrix = design.to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(matrix)
    if rank != matrix.shape[1] or len(subgroup) <= rank:
        raise ValueError("The fixed model needs full rank and residual degrees of freedom.")
    # HC2 divides squared residuals by 1 - leverage; unit leverage is undefined.
    q, _ = np.linalg.qr(matrix, mode="reduced")
    if np.any(1.0 - np.sum(q * q, axis=1) <= 1e-12):
        raise ValueError("HC2 is undefined for observations with unit leverage.")

    result = sm.OLS(subgroup["fin_security_m12"], design).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The fixed test did not produce finite results.")
    if output["std_error"] <= 0:
        raise ValueError("The fixed test needs a positive standard error.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input participant CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    args = parser.parse_args()
    if args.output.resolve() in {args.data.resolve(), Path(__file__).resolve()}:
        parser.error("--output must differ from the input data and this program.")
    try:
        result = analyze(args.data)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
