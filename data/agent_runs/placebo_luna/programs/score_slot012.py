#!/usr/bin/env python3
"""Score the blocked randomized treatment effect on year-two presenteeism."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    outcome = "presenteeism_score_y2"
    required = ["treatment", "block", outcome]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    observed = data[required].dropna()
    treatment = observed["treatment"].astype(float)
    block_indicators = pd.get_dummies(
        observed["block"].astype("category"), drop_first=True, dtype=float
    )
    design = pd.concat([treatment.rename("treatment"), block_indicators], axis=1)
    design = sm.add_constant(design, has_constant="add")

    fit = sm.OLS(observed[outcome].astype(float), design).fit(cov_type="HC2")
    estimate = float(fit.params["treatment"])
    std_error = float(fit.bse["treatment"])
    degrees_freedom = int(fit.nobs - design.shape[1])
    p_value = float(2 * t.sf(abs(estimate / std_error), degrees_freedom))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
