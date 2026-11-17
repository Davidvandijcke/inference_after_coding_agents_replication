#!/usr/bin/env python3
"""Run the frozen Pathways to Work analysis; no outcome/subgroup search.

Estimand: intention-to-treat effect on distress_m12 among parent == 0.
Model: OLS with intercept, treatment, and categorical site fixed effects.
Inference: HC2 standard error; two-sided t test with OLS residual df.
The returned p-value is nominal, without an exploratory-search adjustment.

Requires numpy, pandas, scipy, and statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data: pd.DataFrame) -> dict[str, float]:
    required = ["parent", "site", "treatment", "distress_m12"]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns must be fully observed.")
    for column in ["parent", "treatment"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    # This subgroup and outcome are fixed, regardless of the supplied results.
    sample = data.loc[data["parent"].eq(0), required].copy()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("The parent == 0 subgroup must contain both trial arms.")
    outcome = pd.to_numeric(sample["distress_m12"], errors="raise").astype(float)
    site_indicators = pd.get_dummies(
        sample["site"], prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat(
        [sample[["treatment"]].astype(float), site_indicators], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    matrix = design.to_numpy()
    if not np.isfinite(matrix).all() or not np.isfinite(outcome).all():
        raise ValueError("Analysis values must be finite.")
    if len(sample) <= design.shape[1] or np.linalg.matrix_rank(matrix) < design.shape[1]:
        raise ValueError("The specified model needs a full-rank design and residual df.")

    # Explicit use_t=True is necessary: robust statsmodels fits otherwise use z tests.
    fit = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()) or result["std_error"] <= 0:
        raise ValueError("The specified model did not yield valid HC2 inference.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
