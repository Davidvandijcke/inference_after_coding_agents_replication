#!/usr/bin/env python3
"""Score the exploratory age-by-treatment effect on 12-month self-rated health."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    required = ["site", "treatment", "age", "health_m12"]
    data = pd.read_csv(args.data)
    missing_columns = [name for name in required if name not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    analysis = data[required].copy()
    for name in required:
        analysis[name] = pd.to_numeric(analysis[name], errors="raise")
    analysis = analysis.dropna(subset=required)
    if analysis.empty:
        raise ValueError("No complete observations for the selected analysis")

    # Centering makes the treatment coefficient refer to the sample's mean age;
    # the treatment-by-age interaction is the selected test.
    analysis["age_c"] = analysis["age"] - analysis["age"].mean()
    model = smf.ols(
        "health_m12 ~ treatment * age_c + C(site)", data=analysis
    ).fit(cov_type="HC2", use_t=True)

    term = "treatment:age_c"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }
    output_path = Path(args.output)
    if output_path.resolve() == Path(args.data).resolve():
        raise ValueError("The output path must differ from the input data path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
