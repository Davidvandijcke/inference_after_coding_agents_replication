#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

Estimates the treatment effect on 12-month training enrollment among women,
using OLS with site indicators and HC2 robust standard errors.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with evaluation data")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"female", "treatment", "site", "training_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error(f"data is missing required columns: {', '.join(missing)}")

    women = data.loc[data["female"] == 1]
    if women.empty or women["treatment"].nunique() != 2:
        parser.error("female subgroup must contain observations in both treatment arms")

    result = smf.ols("training_m12 ~ treatment + C(site)", data=women).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
