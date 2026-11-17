#!/usr/bin/env python3
"""Fixed exploratory finding: education=4, distress_m12, site-adjusted ITT.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels.
The two-sided HC2 t-test is nominal; it does not adjust for the original search.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ["education", "distress_m12", "treatment", "site"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns must be fully observed.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must be coded 0 or 1.")

    subgroup = data.loc[data["education"].eq(4)].copy()
    if subgroup["treatment"].nunique() != 2:
        raise ValueError("Education=4 must contain both treatment arms.")
    design = pd.concat(
        [
            pd.Series(1.0, index=subgroup.index, name="const"),
            subgroup[["treatment"]].astype(float),
            pd.get_dummies(subgroup["site"], prefix="site", drop_first=True,
                           dtype=float),
        ],
        axis=1,
    )
    outcome = subgroup["distress_m12"].astype(float)
    if not np.isfinite(outcome.to_numpy()).all():
        raise ValueError("Outcome must be finite.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment and site effects are not separately identified.")
    if len(subgroup) <= design.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")

    result = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    answer = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in answer.values()):
        raise ValueError("The supplied data do not permit finite HC2 inference.")
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must differ from the input dataset.")
    answer = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(answer, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
