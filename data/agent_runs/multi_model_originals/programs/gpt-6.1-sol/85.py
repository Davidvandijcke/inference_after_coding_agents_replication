#!/usr/bin/env python3
"""Run the fixed, exploratory lead analysis from the Pathways to Work packet.

Chosen estimand: intention-to-treat effect on distress_m6, all participants.
Model: distress_m6 ~ treatment + C(site), with an intercept.
Inference: HC2 robust standard error; two-sided t test, residual degrees
of freedom n - rank(X). The p-value is nominal, without a search adjustment.

Usage: python score.py --data <input.csv> --output <result.json>
Requires pandas and statsmodels. No other data files are read.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data):
    required = ["distress_m6", "treatment", "site"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    frame = data[required].copy()
    if frame.isna().any().any():
        raise ValueError("Chosen analysis requires complete outcome, treatment, and site.")
    for column in ("distress_m6", "treatment"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].map(math.isfinite).all():
            raise ValueError(column + " must contain finite numbers.")
    if set(frame.treatment.unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    # C(site) includes a fixed effect for every observed site, irrespective
    # of whether site labels are numeric or strings.
    model = smf.ols("distress_m6 ~ treatment + C(site)", data=frame).fit(
        cov_type="HC2", use_t=True)
    if model.model.rank != model.model.exog.shape[1] or model.df_resid <= 0:
        raise ValueError("The chosen regression is not identifiable on these data.")
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("The chosen regression did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must be separate from the supplied data file.")
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
