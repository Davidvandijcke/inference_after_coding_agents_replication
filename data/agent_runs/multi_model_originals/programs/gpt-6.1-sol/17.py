#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis, without any outcome/group search.

Chosen estimand: the site-adjusted intention-to-treat effect on the 0--20
financial security index at 12 months among participants employed at baseline.

Model, in that subgroup only:
    fin_security_m12 ~ treatment + C(site)

Inference: HC2 standard error and a two-sided Student t test using the OLS
residual degrees of freedom. Outputs nominal, not selection-adjusted, results.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data_path):
    columns = ["site", "treatment", "employed_baseline", "fin_security_m12"]
    data = pd.read_csv(data_path, usecols=columns)
    if data[columns].isna().any().any():
        raise ValueError("Required analysis columns must be fully observed.")
    for col in ["treatment", "employed_baseline", "fin_security_m12"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} contains a non-finite value.")
    for col in ["treatment", "employed_baseline"]:
        if not data[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be coded 0/1.")

    subgroup = data.loc[data.employed_baseline.eq(1)].copy()
    if subgroup.treatment.nunique() != 2:
        raise ValueError("The baseline-employed subgroup needs both treatment arms.")
    model = smf.ols("fin_security_m12 ~ treatment + C(site)", data=subgroup,
                    missing="raise")
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The chosen model's design matrix is not full rank.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("The chosen model needs positive residual degrees of freedom.")
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError("HC2 is undefined for observations with unit leverage.")
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The chosen model produced a non-finite result.")
    if result["std_error"] <= 0:
        raise ValueError("The chosen test needs a positive standard error.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()
    # Guard against accidentally overwriting either supplied packet file.
    output = args.output.resolve()
    protected = {args.data.resolve(), Path(__file__).resolve()}
    protected.update(p.resolve() for p in Path(__file__).resolve().parent.glob("CODEBOOK.md"))
    protected.update(p.resolve() for p in Path(__file__).resolve().parent.glob("data.csv"))
    if output in protected:
        parser.error("The output path must not overwrite an input or the program.")
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
