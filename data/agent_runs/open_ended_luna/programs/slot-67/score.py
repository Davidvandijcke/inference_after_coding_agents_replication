#!/usr/bin/env python3
"""Score the 12-month debt-collections treatment-effect contrast by education."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with the Pathways evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "education", "debt_collections_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("data is missing required columns: " + ", ".join(sorted(missing)))

    # education <= 2 denotes below college; the interaction coefficient is
    # (program effect below college) minus (program effect with some college+).
    data["lower_ed"] = (data["education"] <= 2).astype(int)
    model = smf.ols(
        "debt_collections_m12 ~ treatment * lower_ed + C(site) * lower_ed",
        data=data,
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:lower_ed"

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
