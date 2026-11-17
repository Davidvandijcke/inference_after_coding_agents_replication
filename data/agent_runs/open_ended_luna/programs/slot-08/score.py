#!/usr/bin/env python3
"""Score the age moderation of Pathways to Work's 12-month employment effect."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    if data_path == output_path:
        parser.error("--output must not overwrite the input data")

    data = pd.read_csv(data_path)
    required = {"site", "treatment", "age", "employed_baseline", "employed_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))

    # Center at age 40 and express age in decades. The interaction coefficient
    # is the change in the treatment effect on employment probability per decade.
    data["age_decade_from_40"] = (data["age"] - 40) / 10
    model = smf.ols(
        "employed_m12 ~ treatment * age_decade_from_40 + C(site) + employed_baseline",
        data=data,
    ).fit().get_robustcov_results(cov_type="HC2", use_t=True)

    coefficient = "treatment:age_decade_from_40"
    index = model.model.exog_names.index(coefficient)
    result = {
        "estimate": float(model.params[index]),
        "std_error": float(model.bse[index]),
        "p_value": float(model.pvalues[index]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
