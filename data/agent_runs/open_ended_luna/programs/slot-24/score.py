#!/usr/bin/env python3
"""Score the baseline-employment interaction in six-month financial security."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the site-adjusted treatment-by-baseline-employment "
            "interaction for six-month financial security, using HC2 robust SEs."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_baseline", "fin_security_m6"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    model = smf.ols(
        "fin_security_m6 ~ treatment * employed_baseline + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:employed_baseline"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
