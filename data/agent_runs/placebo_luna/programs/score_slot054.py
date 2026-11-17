#!/usr/bin/env python3
"""Score the prespecified sedentary-job year-2 comparison.

Fits the block-adjusted ANCOVA selected from the supplied evaluation:
sedentary_job_y2 ~ treatment + randomization block indicators
                    + sedentary_job_base
using complete cases and HC2 robust standard errors.
"""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    outcome = "sedentary_job_y2"
    baseline = "sedentary_job_base"
    required = [outcome, baseline, "treatment", "block"]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Keep the same complete-case analysis and randomized-block adjustment.
    sample = data[required].dropna().copy()
    block_indicators = pd.get_dummies(
        sample["block"].astype("category"), drop_first=True, dtype=float
    )
    design = pd.concat(
        [sample[["treatment", baseline]], block_indicators], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    fit = sm.OLS(sample[outcome], design).fit(cov_type="HC2")
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
