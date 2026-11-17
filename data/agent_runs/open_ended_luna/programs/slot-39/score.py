#!/usr/bin/env python3
"""Score the exploratory treatment-by-age effect on 6-month employment.

The reported estimate is the treatment:age coefficient from the fixed model
employed_m6 ~ treatment * age + site indicators, with HC2 robust standard
errors and a two-sided t-test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_m6", "treatment", "age", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))

    result = smf.ols(
        "employed_m6 ~ treatment * age + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:age"
    output = {
        "estimate": float(result.params[term]),
        "std_error": float(result.bse[term]),
        "p_value": float(result.pvalues[term]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
