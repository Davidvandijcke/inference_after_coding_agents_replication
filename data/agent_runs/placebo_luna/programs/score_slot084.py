#!/usr/bin/env python3
"""Score the block-adjusted ITT effect on two-year job satisfaction.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t


OUTCOME = "satisfied_job_y2"
BASELINE = "satisfied_job_base"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"treatment", "block", BASELINE, OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    # Complete-case OLS: outcome on randomized assignment, block indicators,
    # and the corresponding baseline outcome, with HC2 robust standard errors.
    sample = data[["treatment", "block", BASELINE, OUTCOME]].dropna()
    block_terms = pd.get_dummies(
        sample["block"].astype("category"), prefix="block", drop_first=True, dtype=float
    )
    design = pd.concat(
        [sample[["treatment", BASELINE]].astype(float), block_terms], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    result = sm.OLS(sample[OUTCOME].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    estimate = float(result.params["treatment"])
    std_error = float(result.bse["treatment"])
    t_statistic = estimate / std_error
    p_value = float(2 * student_t.sf(abs(t_statistic), result.df_resid))

    output = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
