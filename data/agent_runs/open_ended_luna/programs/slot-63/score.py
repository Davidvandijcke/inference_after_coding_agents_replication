#!/usr/bin/env python3
"""Score the selected treatment-by-female interaction for training at month 12.

The estimate is the difference between the treatment effect for female=1 and
the treatment effect for female=0 in an OLS model with site indicators allowed
to differ across the two groups. The standard error uses HC2 and the two-sided
p-value uses a t reference distribution, as described in CODEBOOK.md.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


REQUIRED_COLUMNS = {"site", "treatment", "female", "training_m12"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        parser.error(f"input CSV is missing required columns: {names}")

    model = smf.ols(
        "training_m12 ~ treatment * female + C(site) + female:C(site)",
        data=data,
        missing="raise",
    ).fit(cov_type="HC2", use_t=True)

    term = "treatment:female"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
