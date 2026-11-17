#!/usr/bin/env python3
"""Score the selected focal analysis from the Pathways evaluation packet.

Fits the site-adjusted OLS model for 12-month food insecurity, allowing the
treatment effect to differ by baseline employment. Reports the interaction
coefficient, its HC2 standard error, and its two-sided t-test p-value.
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
    model = smf.ols(
        "food_insecure_m12 ~ treatment * employed_baseline "
        "+ C(site) * employed_baseline",
        data=data,
    ).fit(cov_type="HC2", use_t=True)

    term = "treatment:employed_baseline"
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
