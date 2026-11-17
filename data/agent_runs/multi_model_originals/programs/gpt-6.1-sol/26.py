#!/usr/bin/env python3
"""Run the frozen, full-sample 12-month financial-security ITT analysis.

Usage: python score.py --data data.csv --output result.json

OLS: fin_security_m12 ~ treatment + site indicators + age + female + parent
     + household_size + employed_baseline + earnings_baseline
     + months_unemployed_baseline + education indicators.

The reported p-value is the nominal, two-sided HC2 t-test of treatment = 0,
with residual degrees of freedom. No outcomes, subgroups, or models are searched.
Requires numpy, pandas, and statsmodels (which depends on scipy).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


BASELINE_COVARIATES = [
    "age", "female", "parent", "household_size", "employed_baseline",
    "earnings_baseline", "months_unemployed_baseline",
]
OUTCOME = "fin_security_m12"


def chosen_analysis(data):
    required = [OUTCOME, "treatment", "site", "education", *BASELINE_COVARIATES]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    values = data[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All analysis variables must be observed and finite.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")

    # Category levels are sorted by pandas; the lowest observed level is omitted.
    design = pd.concat(
        [
            data[["treatment"]],
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
            data[BASELINE_COVARIATES],
            pd.get_dummies(
                data["education"], prefix="education", drop_first=True, dtype=float
            ),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add").astype(float)
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank != design.shape[1] or len(data) <= rank:
        raise ValueError("The fixed model requires full rank and positive residual degrees of freedom.")

    model = sm.OLS(data[OUTCOME].astype(float), design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The fixed model did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    try:
        result = chosen_analysis(pd.read_csv(args.data))
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
