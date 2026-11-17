#!/usr/bin/env python3
"""Score the 12-month training effect difference by degree status.

Fits the preselected OLS model with site indicators and HC2 standard errors,
then writes the two-sided t-test for treatment x bachelor's degree or higher.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data["degree_plus"] = (data["education"] == 4).astype(int)

    model = smf.ols(
        "training_m12 ~ treatment * degree_plus + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)

    term = "treatment:degree_plus"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }
    Path(args.output).write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
