#!/usr/bin/env python3
"""Run the fixed, chosen Pathways to Work analysis.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, statsmodels (and their dependencies).

Estimand: the treatment coefficient for 12-month employment, in probability
units, using all participants. OLS includes site indicators and all eight
baseline covariates; education is categorical, and the remaining covariates
enter linearly. Inference uses HC2 standard errors and a two-sided t-test with
the OLS residual degrees of freedom. No search or multiplicity adjustment is
performed here. See analysis/report.md for the exploratory selection context.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "employed_m12 ~ treatment + C(site) + age + female + parent"
    " + C(education) + household_size + employed_baseline"
    " + earnings_baseline + months_unemployed_baseline"
)
REQUIRED = [
    "employed_m12", "treatment", "site", "age", "female", "parent",
    "education", "household_size", "employed_baseline", "earnings_baseline",
    "months_unemployed_baseline",
]


def score(data_path):
    data = pd.read_csv(data_path)
    missing = sorted(set(REQUIRED).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    numeric = data[REQUIRED].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Required columns must contain finite, observed values.")
    for column in ["employed_m12", "treatment", "female", "parent", "employed_baseline"]:
        if not numeric[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be coded 0 or 1.")
    if set(numeric.treatment.unique()) != {0, 1}:
        raise ValueError("Both treatment and control participants are required.")
    if not numeric.site.isin([1, 2, 3, 4]).all():
        raise ValueError("site must be coded 1 through 4.")
    if not numeric.education.isin([1, 2, 3, 4]).all():
        raise ValueError("education must be coded 1 through 4.")

    model = smf.ols(FORMULA, data=numeric, missing="raise")
    design = model.exog
    treatment_index = model.exog_names.index("treatment")
    nuisance = np.delete(design, treatment_index, axis=1)
    rank = np.linalg.matrix_rank(design)
    if rank <= np.linalg.matrix_rank(nuisance):
        raise ValueError("The treatment effect is not identifiable in this dataset.")
    if len(numeric) <= rank:
        raise ValueError("Positive residual degrees of freedom are required.")
    result = model.fit(cov_type="HC2", use_t=True)
    values = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("The model did not produce finite estimates and inference.")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    values = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
