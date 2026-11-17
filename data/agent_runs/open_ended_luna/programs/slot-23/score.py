#!/usr/bin/env python3
"""Score the exploratory gender difference in the 6-month health effect.

Fits site-adjusted OLS with site-specific female intercepts and a
treatment-by-female interaction. The reported estimate is the interaction:
the treatment effect for women minus the treatment effect for men. HC2
heteroskedasticity-robust standard errors and a two-sided t-test are used.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t


FORMULA = "health_m6 ~ treatment * female + C(site) * female"
TERM = "treatment:female"
REQUIRED_COLUMNS = {"health_m6", "treatment", "female", "site"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    result = smf.ols(FORMULA, data=data).fit(cov_type="HC2")
    estimate = float(result.params[TERM])
    std_error = float(result.bse[TERM])
    statistic = estimate / std_error
    p_value = float(2 * student_t.sf(abs(statistic), df=result.df_resid))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "estimate": estimate,
                "std_error": std_error,
                "p_value": p_value,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
