#!/usr/bin/env python3
"""Frozen Pathways to Work analysis; no outcome or subgroup search.

Test: assignment effect on six-month employment among participants with
employed_baseline == 0. Fit employed_m6 ~ treatment + C(site) using OLS,
HC2 covariance, and a two-sided Student t test with residual degrees of freedom.
The estimate and standard error are in probability units (0.01 = 1 pp).

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"employed_baseline", "employed_m6", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if data["employed_baseline"].isna().any():
        raise ValueError("employed_baseline must be observed for every participant")
    frame = data.loc[data["employed_baseline"].eq(0)].copy()
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Selected subgroup must be nonempty with complete analysis variables")
    if set(frame["treatment"].unique()) != {0, 1}:
        raise ValueError("Selected subgroup must contain both treatment arms, coded 0 and 1")
    if not frame["employed_m6"].isin([0, 1]).all():
        raise ValueError("employed_m6 must be coded 0 or 1")
    model = smf.ols("employed_m6 ~ treatment + C(site)", data=frame)
    design = model.exog
    if np.linalg.matrix_rank(design) < design.shape[1] or len(frame) <= design.shape[1]:
        raise ValueError("Insufficient independent variation for the chosen regression")
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Chosen regression produced a nonfinite result")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
