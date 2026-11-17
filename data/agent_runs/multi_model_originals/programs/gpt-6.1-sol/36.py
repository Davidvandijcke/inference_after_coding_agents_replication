#!/usr/bin/env python3
"""Run the fixed, site-adjusted intention-to-treat analysis of hours change.

Outcome: hours_m12 - hours_m6, including zero hours for nonworkers.
Model: OLS with intercept, treatment, and categorical site indicators.
Inference: HC2 standard error; two-sided Student t test with OLS residual df.
Dependencies: numpy, pandas, statsmodels.
No outcome, subgroup, time-point, or covariate selection is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    columns = ["site", "treatment", "hours_m6", "hours_m12"]
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or data[columns].isna().any().any():
        raise ValueError("The analysis requires nonempty data with no missing analysis values.")
    numeric = data[["treatment", "hours_m6", "hours_m12"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Treatment and hours must be finite numeric values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")

    design = pd.concat(
        [
            data[["treatment"]],
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add").astype(float)
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The treatment/site model requires full rank and positive residual degrees of freedom.")
    outcome = data["hours_m12"] - data["hours_m6"]
    model = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The supplied data do not yield finite HC2 inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    args = parser.parse_args()
    result = score(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
