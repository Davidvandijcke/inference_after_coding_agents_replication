#!/usr/bin/env python3
"""Score one fixed analysis, selected in the Pathways to Work exploration.

Population: participants with parent == 1 at baseline.
Outcome: training_m12 (0/1).
Model: OLS with an intercept, treatment, and site indicators.
Inference: HC2 standard error and a two-sided t test using residual df.
Effect units: probability units (multiply by 100 for percentage points).

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
This program neither searches specifications nor corrects for prior selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data: pd.DataFrame) -> dict[str, float]:
    required = ["parent", "training_m12", "treatment", "site"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data["parent"].isna().any() or not data["parent"].isin([0, 1]).all():
        raise ValueError("parent must be observed and coded 0/1.")

    sample = data.loc[data["parent"].eq(1), required].reset_index(drop=True)
    if sample.empty:
        raise ValueError("No participants have parent == 1.")
    if sample.isna().any().any():
        raise ValueError("The selected analysis requires complete observations.")
    for name in ["training_m12", "treatment"]:
        if not sample[name].isin([0, 1]).all():
            raise ValueError(f"{name} must be coded 0/1.")
    if sample["treatment"].nunique() != 2:
        raise ValueError("Both treatment groups are required among parents.")

    sites = pd.get_dummies(sample["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = pd.concat([
        pd.Series(1.0, index=sample.index, name="intercept"),
        sample["treatment"].astype(float), sites,
    ], axis=1)
    if len(sample) <= design.shape[1]:
        raise ValueError("Too few observations for residual degrees of freedom.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment is not identified after controlling for site.")

    fit = sm.OLS(sample["training_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The selected test has undefined inference on this dataset.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = score(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
