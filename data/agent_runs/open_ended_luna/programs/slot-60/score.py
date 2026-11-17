#!/usr/bin/env python3
"""Score the preselected six-month training-effect contrast by education."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def subgroup_effect(data: pd.DataFrame, education_level: int):
    """Site-adjusted treatment effect and HC2 variance in one subgroup."""
    subgroup = data.loc[data["education"] == education_level]
    if subgroup.empty:
        raise ValueError(f"No participants with education={education_level}")
    if subgroup["treatment"].nunique() != 2:
        raise ValueError(
            f"Education={education_level} must contain both treatment arms"
        )

    result = smf.ols(
        "training_m6 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2")
    return (
        float(result.params["treatment"]),
        float(result.bse["treatment"] ** 2),
        float(result.df_resid),
    )


def score(data_path: Path):
    data = pd.read_csv(data_path)
    required = {"education", "training_m6", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    # Compare the site-adjusted ITT effect for high-school/GED participants
    # (education=2) with the effect for participants without a high-school
    # credential (education=1). The disjoint subgroup estimates are independent.
    effect_less_hs, var_less_hs, df_less_hs = subgroup_effect(data, 1)
    effect_hs_ged, var_hs_ged, df_hs_ged = subgroup_effect(data, 2)

    estimate = effect_hs_ged - effect_less_hs
    variance = var_hs_ged + var_less_hs
    std_error = math.sqrt(variance)
    df = variance**2 / (
        var_hs_ged**2 / df_hs_ged + var_less_hs**2 / df_less_hs
    )
    t_stat = estimate / std_error
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": float(p_value),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the preselected education-group contrast."
    )
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
