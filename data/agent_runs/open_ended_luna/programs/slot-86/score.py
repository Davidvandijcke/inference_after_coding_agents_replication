#!/usr/bin/env python3
"""Score the prespecified follow-up analysis: treatment-by-sex interaction
for 12-month psychological distress, adjusting for randomization site.

Usage:
    python score.py --data <path> --output <path>
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

    data_path = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    if data_path == output_path:
        parser.error("--output must not overwrite the input data")

    data = pd.read_csv(data_path)
    required = {"distress_m12", "treatment", "female", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("Missing required columns: " + ", ".join(missing))

    # The interaction coefficient is the difference in the treatment effect
    # for women versus men. HC2 robust standard errors and a two-sided t-test
    # follow the evaluation's analysis convention.
    result = smf.ols(
        "distress_m12 ~ treatment * female + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:female"
    payload = {
        "estimate": float(result.params[term]),
        "std_error": float(result.bse[term]),
        "p_value": float(result.pvalues[term]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
