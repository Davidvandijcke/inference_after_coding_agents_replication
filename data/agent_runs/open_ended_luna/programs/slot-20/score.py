#!/usr/bin/env python3
"""Score the site-adjusted six-month employment effect in the education=2 subgroup."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the treatment effect on six-month employment among "
            "participants with a high school diploma or GED."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["education"] == 2]
    result = smf.ols(
        "employed_m6 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
