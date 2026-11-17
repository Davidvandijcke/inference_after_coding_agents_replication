#!/usr/bin/env python3
"""Run the fixed, exploratory-selected Pathways to Work analysis.

Chosen estimand: intention-to-treat effect on health_m6 in all participants.
Model: health_m6 ~ treatment + categorical site, with an intercept.
Inference: HC2 robust standard error and two-sided t test, residual degrees
of freedom. The p-value is nominal; it does not adjust for the original search.

Dependencies: numpy, pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data: pd.DataFrame) -> dict[str, float]:
    """Fit only the chosen model; do not select outcomes or participant groups."""
    required = ["health_m6", "treatment", "site"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    sample = data.loc[:, required].copy()
    if sample.empty or sample.isna().any().any():
        raise ValueError("The chosen analysis requires nonmissing observations.")
    for column in ["health_m6", "treatment"]:
        sample[column] = pd.to_numeric(sample[column], errors="raise")
        if not np.isfinite(sample[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values.")
    if set(sample["treatment"].unique()) != {0, 1}:
        raise ValueError("treatment must contain both 0 and 1, and no other values.")
    if not sample["health_m6"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("health_m6 must use the codebook's 1-5 scale.")

    model = smf.ols("health_m6 ~ treatment + C(site)", data=sample,
                    missing="raise")
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The treatment effect is not identifiable from these data.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1.0 - 1e-12):
        raise ValueError("HC2 is undefined for an observation with unit leverage.")

    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The chosen model produced nonfinite inference.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
