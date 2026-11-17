#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the site-adjusted treatment effect on 12-month debt "
            "collections among participants with some college or more."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["education"] >= 3]
    result = smf.ols(
        "debt_collections_m12 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
