#!/usr/bin/env python3
"""Score the prespecified-for-this-script year-one other-tobacco contrast.

Fits outcome ~ treatment + randomization-block indicators + baseline outcome,
using OLS with HC2 robust standard errors and a two-sided t-test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["other_tobacco_y1", "other_tobacco_base", "treatment", "block"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    analysis_data = data[required].dropna()
    model = smf.ols(
        "other_tobacco_y1 ~ treatment + C(block) + other_tobacco_base",
        data=analysis_data,
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
