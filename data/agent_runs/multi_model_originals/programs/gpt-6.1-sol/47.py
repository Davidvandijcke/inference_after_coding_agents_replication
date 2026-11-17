#!/usr/bin/env python3
"""Run the fixed full-sample, 12-month earnings intention-to-treat analysis.

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json

Model: earnings_m12 ~ treatment + C(site), with an intercept.
Inference: HC2 standard error and a two-sided Student t test, using residual
degrees of freedom. No subgroup selection, covariate selection, or search.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path: Path) -> dict[str, float]:
    columns = ["earnings_m12", "treatment", "site"]
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError("The analysis columns must contain finite, observed values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Both treatment=0 and treatment=1 must be present.")

    model = smf.ols("earnings_m12 ~ treatment + C(site)", data=data, missing="raise")
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment must be identifiable after accounting for site.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The data do not support finite HC2 inference.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
