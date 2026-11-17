#!/usr/bin/env python3
"""Score the 12-month financial-security treatment-effect contrast by baseline employment."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {
        "site",
        "treatment",
        "employed_baseline",
        "fin_security_m12",
    }
    missing = required.difference(data.columns)
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(sorted(missing)))

    # Separate site intercepts by baseline-employment group reproduce the
    # codebook's site-adjusted subgroup regressions. The interaction coefficient
    # is the difference between the two treatment effects.
    result = smf.ols(
        "fin_security_m12 ~ treatment * employed_baseline + C(site) * employed_baseline",
        data=data,
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:employed_baseline"

    payload = {
        "estimate": float(result.params[term]),
        "std_error": float(result.bse[term]),
        "p_value": float(result.pvalues[term]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
