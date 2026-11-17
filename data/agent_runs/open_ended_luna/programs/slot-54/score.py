#!/usr/bin/env python3
"""Score the education-by-treatment interaction for six-month distress.

The estimate is the difference in site-adjusted treatment effects between
participants with some college or more and those with high school/GED or less.
Inference uses HC2 standard errors and a two-sided t-test.

Requires pandas and statsmodels.
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
    required = {"site", "treatment", "education", "distress_m6"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    data["education_3plus"] = (data["education"] >= 3).astype(int)
    model = smf.ols(
        "distress_m6 ~ treatment * education_3plus + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)

    term = "treatment:education_3plus"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
