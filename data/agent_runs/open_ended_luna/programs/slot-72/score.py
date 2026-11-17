#!/usr/bin/env python3
"""Score the exploratory Pathways to Work finding in baseline parents.

Fits OLS of 12-month training enrollment on treatment and site indicators,
restricted to participants who lived with a child at baseline. Uses HC2 robust
standard errors and the two-sided t test specified in CODEBOOK.md.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"parent", "training_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(sorted(missing)))

    parents = data.loc[data["parent"] == 1]
    if parents.empty:
        parser.error("input CSV has no participants with parent == 1")

    model = smf.ols("training_m12 ~ treatment + C(site)", data=parents).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
