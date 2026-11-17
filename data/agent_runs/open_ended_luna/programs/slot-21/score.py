#!/usr/bin/env python3
"""Score the prespecified follow-up hypothesis on a Pathways-style dataset.

The sole test is whether the treatment effect on 12-month financial security
differs by baseline employment status. The treatment-by-employment coefficient
is estimated with site fixed effects within each employment stratum, HC2 robust
standard errors, and a two-sided t test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "fin_security_m12 ~ treatment * employed_baseline "
    "+ C(site) * employed_baseline"
)
INTERACTION = "treatment:employed_baseline"
REQUIRED_COLUMNS = {
    "fin_security_m12",
    "treatment",
    "employed_baseline",
    "site",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Input is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")
    for column in ("treatment", "employed_baseline"):
        if not set(data[column].unique()).issubset({0, 1}):
            raise ValueError(f"{column} must contain only 0 and 1")

    result = smf.ols(FORMULA, data=data).fit(cov_type="HC2", use_t=True)
    if INTERACTION not in result.params:
        raise ValueError("Could not estimate the treatment-by-employment contrast")

    payload = {
        "estimate": float(result.params[INTERACTION]),
        "std_error": float(result.bse[INTERACTION]),
        "p_value": float(result.pvalues[INTERACTION]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
