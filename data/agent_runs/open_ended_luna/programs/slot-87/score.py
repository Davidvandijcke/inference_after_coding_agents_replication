#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup contrast.

The analysis compares the site-adjusted treatment effect on six-month self-rated
health between participants unemployed for at least 12 months at baseline and
all other participants. It uses HC2 standard errors within each subgroup and a
two-sided Welch-Satterthwaite t test for the difference in effects.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def treatment_effect(frame: pd.DataFrame) -> tuple[float, float, float]:
    """Return the treatment coefficient, HC2 variance, and residual df."""
    if frame.empty or frame["treatment"].nunique() != 2:
        raise ValueError("Each baseline-duration subgroup must contain both treatment arms.")

    fit = smf.ols("health_m6 ~ treatment + C(site)", data=frame).fit(
        cov_type="HC2", use_t=True
    )
    return (
        float(fit.params["treatment"]),
        float(fit.bse["treatment"] ** 2),
        float(fit.df_resid),
    )


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "months_unemployed_baseline", "health_m6"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values.")

    long_unemployed = data["months_unemployed_baseline"] >= 12
    long_effect, long_var, long_df = treatment_effect(data.loc[long_unemployed])
    other_effect, other_var, other_df = treatment_effect(data.loc[~long_unemployed])

    estimate = long_effect - other_effect
    variance = long_var + other_var  # The two subgroup samples do not overlap.
    std_error = variance**0.5
    df = variance**2 / (long_var**2 / long_df + other_var**2 / other_df)
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df))

    return {
        "estimate": float(estimate),
        "std_error": float(std_error),
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    if args.data.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the input data file")

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
