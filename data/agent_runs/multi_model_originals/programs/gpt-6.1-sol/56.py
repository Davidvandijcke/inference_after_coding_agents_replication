#!/usr/bin/env python3
"""Run the single, frozen Pathways to Work analysis.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels (which also requires scipy).

Estimand: the site-adjusted intention-to-treat effect on earnings_m12, in
dollars earned in the previous month, among participants not employed at
baseline with months_unemployed_baseline >= 12. Fit an intercept, treatment,
and indicators for observed sites (omit one reference site). Use HC2 standard
errors and a two-sided residual-degrees-of-freedom t test. No covariate,
subgroup, outcome, or follow-up search is performed here. The p value is
nominal, without adjustment for the exploration that motivated this choice.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data: pd.DataFrame) -> dict:
    required = ["site", "treatment", "employed_baseline",
                "months_unemployed_baseline", "earnings_m12"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns must have no missing values.")
    if not data.employed_baseline.isin([0, 1]).all():
        raise ValueError("employed_baseline must be coded 0/1.")
    sample = data.loc[(data.employed_baseline == 0) &
                      (data.months_unemployed_baseline >= 12)].copy()
    if set(sample.treatment.unique()) != {0, 1}:
        raise ValueError("The fixed subgroup must contain both treatment arms, coded 0/1.")
    design = sm.add_constant(pd.concat([
        sample[["treatment"]].astype(float),
        pd.get_dummies(sample.site, prefix="site", drop_first=True, dtype=float)
    ], axis=1), has_constant="add")
    x = design.to_numpy(dtype=float)
    y = sample.earnings_m12.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Analysis values must be finite.")
    if len(sample) <= x.shape[1] or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("The fixed subgroup has insufficient variation for this model.")
    fit = sm.OLS(y, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()) or result["std_error"] <= 0:
        raise ValueError("HC2 inference is undefined for this dataset and fixed subgroup.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
