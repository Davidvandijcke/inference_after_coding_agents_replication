#!/usr/bin/env python3
"""Run the frozen Site 1, 12-month financial-security analysis.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels (which depends on scipy).

Estimand: the intention-to-treat effect of assignment on fin_security_m12
among participants at site 1. OLS includes an intercept and treatment.
Site indicators are redundant because this subgroup contains one site.
Inference uses HC2 standard errors and a two-sided Student t test with
OLS residual degrees of freedom. The p-value is nominal: it does not
adjust for the exploratory search that originally selected this analysis.
No outcome, subgroup, or model selection is performed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "fin_security_m12"])
    data["site"] = pd.to_numeric(data["site"], errors="raise")
    subgroup = data.loc[data["site"].eq(1)].copy()
    for column in ("treatment", "fin_security_m12"):
        subgroup[column] = pd.to_numeric(subgroup[column], errors="raise")
    if not np.isfinite(subgroup.to_numpy(dtype=float)).all():
        raise ValueError("Site 1 analysis variables must be finite and observed.")
    if not subgroup["treatment"].isin([0, 1]).all():
        raise ValueError("Site 1 treatment values must be 0 or 1.")
    counts = subgroup["treatment"].value_counts()
    if any(counts.get(arm, 0) < 2 for arm in (0, 1)):
        raise ValueError("Site 1 must contain at least two participants per arm.")

    design = sm.add_constant(subgroup[["treatment"]].astype(float), has_constant="add")
    result = sm.OLS(subgroup["fin_security_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    answer = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in answer.values()):
        raise ValueError("The chosen test is undefined for this dataset.")
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    answer = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(answer, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
