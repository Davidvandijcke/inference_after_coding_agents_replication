#!/usr/bin/env python3
"""Score the exploratory some-college treatment effect in a supplied dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["education"] == 3]
    model = smf.ols("effect_per_unit ~ treatment + C(block)", data=subgroup).fit(
        cov_type="HC2"
    )
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    degrees_freedom = float(model.df_resid)
    p_value = float(2 * t.sf(abs(estimate / std_error), degrees_freedom))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
