#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis, without any exploratory search.

Chosen estimand: the intention-to-treat effect on six-month food insecurity
among all participants. Model: food_insecure_m6 ~ treatment + C(site), with
an intercept, HC2 standard errors, and a two-sided Student t test using the
OLS residual degrees of freedom. The estimate and standard error are in
probability units (multiply by 100 for percentage points).

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data: pd.DataFrame) -> dict[str, float]:
    """Estimate only the fixed, full-sample, six-month outcome regression."""
    required = ["food_insecure_m6", "treatment", "site"]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    sample = data.loc[:, required].copy()
    if sample.empty or sample.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    for name in ["food_insecure_m6", "treatment"]:
        sample[name] = pd.to_numeric(sample[name], errors="raise")
        if not sample[name].isin([0, 1]).all():
            raise ValueError(f"{name} must contain only 0 and 1.")
    if sample.treatment.nunique() != 2:
        raise ValueError("Both randomized treatment arms must be present.")

    model = smf.ols(
        "food_insecure_m6 ~ treatment + C(site)", data=sample,
        missing="raise",
    )
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment must be identifiable after controlling for site.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(1 - leverage <= np.finfo(float).eps * 10):
        raise ValueError("HC2 is undefined for an observation with leverage one.")

    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The fitted model did not produce finite inference.")
    if result["std_error"] <= 0:
        raise ValueError("The treatment standard error must be positive.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON")
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    try:
        result = chosen_analysis(pd.read_csv(args.data))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
