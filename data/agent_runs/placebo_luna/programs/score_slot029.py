#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on capped effective WTP."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with evaluation data")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"treatment", "block", "wtp_effective_capped"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = data.dropna(subset=["treatment", "block", "wtp_effective_capped"])
    # Intercept, treatment indicator, and block fixed effects (first block omitted).
    blocks = pd.get_dummies(data["block"], prefix="block", drop_first=True, dtype=float)
    design = pd.concat(
        [pd.Series(1.0, index=data.index, name="intercept"),
         data["treatment"].astype(float).rename("treatment"), blocks],
        axis=1,
    )
    result = sm.OLS(data["wtp_effective_capped"].astype(float), design).fit()
    robust = result.get_robustcov_results(cov_type="HC2", use_t=True)
    treatment_index = design.columns.get_loc("treatment")
    estimate = float(robust.params[treatment_index])
    std_error = float(robust.bse[treatment_index])
    p_value = float(2 * t.sf(abs(estimate / std_error), df=result.df_resid))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
