#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month psychological distress."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "distress_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))
    if data[list(required)].isna().any().any():
        parser.error("site, treatment, and distress_m12 must be complete")

    # Site indicators account for the within-site randomization. HC2 supplies
    # heteroskedasticity-robust standard errors; use_t selects a two-sided t-test.
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    design = pd.concat(
        [data[["treatment"]].astype(float).reset_index(drop=True),
         site_indicators.reset_index(drop=True)],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(data["distress_m12"].astype(float).reset_index(drop=True), design)
    result = model.fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
