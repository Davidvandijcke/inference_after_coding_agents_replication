#!/usr/bin/env python3
"""Score the randomized module's effect on perceived market offsets.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t


OUTCOME = "effect_below_one_to_one"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = [OUTCOME, "treatment", "block"]
    missing = [column for column in needed if column not in data.columns]
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))

    analysis_data = data.dropna(subset=needed)
    if analysis_data.empty:
        parser.error("input has no complete observations for the chosen analysis")

    fit = smf.ols(f"{OUTCOME} ~ treatment + C(block)", data=analysis_data).fit(
        cov_type="HC2", use_t=True
    )
    estimate = float(fit.params["treatment"])
    std_error = float(fit.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * student_t.sf(abs(statistic), df=fit.df_resid))

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
