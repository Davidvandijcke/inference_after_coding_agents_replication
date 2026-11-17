#!/usr/bin/env python3
"""Score the six-month distress treatment-effect difference by sex.

Fits the fixed interaction model selected from the supplied
evaluation: distress_m6 on treatment, female, their interaction, and site fixed
effects. The interaction estimate is the treatment effect for women minus the
treatment effect for men. Inference uses HC2 standard errors and a two-sided
t-test with the model residual degrees of freedom.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "female", "distress_m6"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))
    if data[list(required)].isna().any().any():
        parser.error("required analysis columns must not contain missing values")

    model = smf.ols(
        "distress_m6 ~ treatment * female + C(site)", data=data
    ).fit()
    robust = model.get_robustcov_results(cov_type="HC2")
    interaction = list(model.params.index).index("treatment:female")
    estimate = float(model.params.iloc[interaction])
    std_error = float(robust.bse[interaction])
    p_value = float(
        2 * t.sf(abs(estimate / std_error), df=float(model.df_resid))
    )

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
