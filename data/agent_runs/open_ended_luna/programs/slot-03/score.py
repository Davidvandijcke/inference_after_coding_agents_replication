#!/usr/bin/env python3
"""Score the prespecified parent-by-treatment interaction for 6-month collections."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV with evaluation records")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "parent", "debt_collections_m6"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))

    # Let parent status have site-specific means so the interaction compares
    # the two site-adjusted subgroup treatment effects.
    model = smf.ols(
        "debt_collections_m6 ~ treatment * parent + C(site) * parent", data=data
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:parent"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
