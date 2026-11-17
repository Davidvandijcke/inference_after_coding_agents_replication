#!/usr/bin/env python3
"""Score the fixed 12-month employment analysis for a Pathways dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "employed_m12 ~ treatment + C(site) + age + female + parent "
    "+ C(education) + household_size + employed_baseline "
    "+ earnings_baseline + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    fit = smf.ols(FORMULA, data=data).fit(cov_type="HC2")
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
