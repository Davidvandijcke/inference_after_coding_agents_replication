#!/usr/bin/env python3
"""Rerun the selected contrast for the Pathways evaluation.

The contrast is the 12-month training-enrollment treatment effect for
education=2 minus the effect for education=1. Each education group has its own
site indicators. Inference uses HC2 standard errors and a two-sided t test.
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
    required = {"site", "treatment", "education", "training_m12"}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data = data.loc[data["education"].isin([1, 2]), list(required)].copy()
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Both treatment assignments must be present")
    if set(data["education"].unique()) != {1, 2}:
        raise ValueError("Education categories 1 and 2 must both be present")

    # Keep the education reference category fixed so the interaction coefficient
    # is always the education=2 effect minus the education=1 effect.
    data["education"] = pd.Categorical(data["education"], categories=[1, 2])
    cell_counts = data.groupby(["site", "education", "treatment"], observed=False).size()
    if (cell_counts == 0).any():
        raise ValueError(
            "Each site, education (1 or 2), and treatment cell must be represented"
        )

    model = smf.ols(
        "training_m12 ~ treatment * C(education) + C(site) * C(education)",
        data=data,
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:C(education)[T.2]"
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
