#!/usr/bin/env python3
"""Score the preselected gender-group interaction for 12-month debt collections."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "debt_collections_m12 ~ treatment * C(female) + C(site) * C(female)"
)
TERM = "treatment:C(female)[T.1]"
REQUIRED_COLUMNS = {
    "debt_collections_m12",
    "treatment",
    "female",
    "site",
}


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the selected treatment-by-female interaction."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    if data_path.resolve() == output_path.resolve():
        parser.error("--output must not overwrite the input data")

    data = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        parser.error("input is missing required columns: " + ", ".join(sorted(missing)))

    fit = smf.ols(FORMULA, data=data).fit(cov_type="HC2", use_t=True)
    if TERM not in fit.params:
        parser.error("cannot estimate the interaction; both female groups are required")

    result = {
        "estimate": float(fit.params[TERM]),
        "std_error": float(fit.bse[TERM]),
        "p_value": float(fit.pvalues[TERM]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
