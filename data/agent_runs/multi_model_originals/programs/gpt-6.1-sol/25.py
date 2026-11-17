#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis selected in report.md.

Usage: python score.py --data data.csv --output result.json

Analysis: among participants age < 35 at baseline, regress training_m6 on
an intercept, randomized treatment, and site fixed effects. Report the
treatment coefficient in probability units, its HC2 standard error, and
the two-sided t-test p-value with n - rank(X) degrees of freedom.

Requires numpy, pandas, and statsmodels. No exploration or model selection
is performed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    data = pd.read_csv(data_path)
    required = ["age", "treatment", "site", "training_m6"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    data = data[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Analysis columns must be observed and finite.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must be coded 0/1.")
    if not data["training_m6"].isin([0, 1]).all():
        raise ValueError("training_m6 must be coded 0/1.")

    subgroup = data.loc[data["age"] < 35].copy()
    if subgroup["treatment"].nunique() != 2:
        raise ValueError("The under-35 subgroup must contain both treatment arms.")
    site_indicators = pd.get_dummies(
        subgroup["site"], prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat(
        [pd.Series(1.0, index=subgroup.index, name="intercept"),
         subgroup["treatment"], site_indicators], axis=1
    )
    matrix = design.to_numpy(dtype=float)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("Treatment is not separately estimable from site effects.")
    if len(subgroup) <= matrix.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")
    result = sm.OLS(subgroup["training_m6"], design).fit(
        cov_type="HC2", use_t=True
    )
    if np.any(result.get_influence().hat_matrix_diag >= 1 - 1e-12):
        raise ValueError("Unit leverage prevents HC2 inference.")
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The selected analysis produced undefined inference.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must differ from the input dataset.")
    try:
        output = chosen_analysis(args.data)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
