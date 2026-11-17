#!/usr/bin/env python3
"""Frozen Pathways to Work analysis; no outcome or subgroup search.

Outcome: earnings_m6, in dollars earned in the previous month.
Population: employed_baseline == 0 and months_unemployed_baseline >= 6.
Model: OLS with an intercept, treatment, and site indicators.
Inference: HC2 standard error; two-sided Student t with residual degrees
of freedom, as specified in CODEBOOK.md. Reported p-value is unadjusted.

Requires pandas, numpy, scipy, and statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data: pd.DataFrame) -> dict[str, float]:
    required = [
        "site", "treatment", "employed_baseline",
        "months_unemployed_baseline", "earnings_m6",
    ]
    absent = set(required) - set(data.columns)
    if absent:
        raise ValueError(f"Missing required columns: {sorted(absent)}")
    values = data[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Required columns must contain finite, observed values.")
    for column in ["treatment", "employed_baseline"]:
        if not values[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be coded 0 or 1.")

    sample = values.loc[
        (values["employed_baseline"] == 0)
        & (values["months_unemployed_baseline"] >= 6)
    ]
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("Chosen subgroup must contain both treatment arms.")
    design = sm.add_constant(
        pd.concat(
            [sample[["treatment"]],
             pd.get_dummies(sample["site"], prefix="site", drop_first=True, dtype=float)],
            axis=1,
        ),
        has_constant="add",
    )
    matrix = design.to_numpy(dtype=float)
    if len(sample) <= design.shape[1] or np.linalg.matrix_rank(matrix) < design.shape[1]:
        raise ValueError("Chosen regression needs full rank and positive residual degrees of freedom.")
    model = sm.OLS(sample["earnings_m6"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The chosen test has undefined inference on this dataset.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output must differ from the input data path.")
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
