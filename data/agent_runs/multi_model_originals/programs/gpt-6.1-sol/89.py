#!/usr/bin/env python3
"""Run the fixed, exploratory-selected Pathways to Work analysis.

Chosen test: six-month financial security, participants at site 3, intention
to treat. OLS on an intercept and treatment, HC2 covariance, two-sided t test
with residual degrees of freedom. Site indicators are constant in this subgroup
and therefore absorbed by the intercept. No baseline covariates, searching,
multiple-testing correction, or alternative-model selection is performed.

Dependencies: numpy, pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data: pd.DataFrame) -> dict[str, float]:
    required = {"site", "treatment", "fin_security_m6"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    site = pd.to_numeric(data["site"], errors="raise")
    subset = data.loc[site.eq(3), ["treatment", "fin_security_m6"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(subset.to_numpy(dtype=float)).all():
        raise ValueError("Site 3 treatment and outcome values must be finite.")
    if not subset["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must be coded 0 or 1.")
    if any((subset["treatment"] == arm).sum() < 2 for arm in (0, 1)):
        raise ValueError("Site 3 must have at least two participants in each arm.")
    design = sm.add_constant(subset[["treatment"]].astype(float), has_constant="add")
    fit = sm.OLS(subset["fin_security_m6"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The selected regression produced undefined inference.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output must be separate from the input data.")
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
