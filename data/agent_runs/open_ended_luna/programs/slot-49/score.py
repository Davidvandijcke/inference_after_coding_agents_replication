#!/usr/bin/env python3
"""Score the female-minus-male treatment-effect difference in 12-month debt collections."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "female", "debt_collections_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    # The female-by-site terms allow site-specific baselines within each gender,
    # matching the codebook's site-adjusted subgroup analyses. The treatment:female
    # coefficient is the female treatment effect minus the male treatment effect.
    result = smf.ols(
        "debt_collections_m12 ~ treatment * female + C(site) * female",
        data=data,
    ).fit(cov_type="HC2", use_t=True)

    payload = {
        "estimate": float(result.params["treatment:female"]),
        "std_error": float(result.bse["treatment:female"]),
        "p_value": float(result.pvalues["treatment:female"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
