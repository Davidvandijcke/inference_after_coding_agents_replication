#!/usr/bin/env python3
"""Score the prespecified exploratory heterogeneity analysis.

Usage: python score.py --data <path> --output <path>

The test is the treatment-by-parent interaction for food_insecure_m6 in an
OLS model with site indicators and site-by-parent terms. This allows the
parent/nonparent outcome gap to vary by site while estimating a common
treatment effect within each group. Inference uses HC2 standard errors and a
two-sided t-test, as specified in CODEBOOK.md.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"food_insecure_m6", "treatment", "parent", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    model = smf.ols(
        "food_insecure_m6 ~ treatment * parent + C(site) * parent", data=data
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
