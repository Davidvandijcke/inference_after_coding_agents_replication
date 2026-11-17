#!/usr/bin/env python3
"""Run the selected Pathways to Work analysis on a compatible CSV.

Selected analysis: regress 6-month financial security on treatment and site
indicators, with HC2 robust standard errors and a two-sided t-test.
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
    required = {"site", "treatment", "fin_security_m6"}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        parser.error("missing required columns: " + ", ".join(missing_columns))
    if data[list(required)].isna().any().any():
        parser.error("site, treatment, and fin_security_m6 must be complete")

    result = smf.ols(
        "fin_security_m6 ~ treatment + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)

    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
