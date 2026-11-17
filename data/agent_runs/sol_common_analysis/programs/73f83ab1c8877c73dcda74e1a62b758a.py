#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment, adjusting for site.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, statsmodels.
The reported p-value is the unadjusted, two-sided HC2 t-test for treatment.
No other outcomes, subgroups, or specifications are evaluated.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete observations.")
    for column in ("employed_m12", "treatment"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
        raise ValueError("Treatment must be identifiable after adjusting for site.")
    if model.exog.shape[0] <= model.exog.shape[1]:
        raise ValueError("The model requires positive residual degrees of freedom.")

    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The specified HC2 t-test is not estimable on these data.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = score(args.data)
    args.output.write_text(json.dumps(output, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
