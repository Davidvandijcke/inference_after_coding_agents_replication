#!/usr/bin/env python3
"""Score the selected subgroup result for Pathways to Work.

Fits financial security at month 12 on treatment and site fixed effects among
participants coded female=0, using HC2 robust standard errors and a two-sided
t-test for the treatment coefficient.
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
    required = {"female", "treatment", "site", "fin_security_m12"}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        parser.error(f"missing required columns: {', '.join(missing_columns)}")

    sample = data.loc[data["female"].eq(0), ["fin_security_m12", "treatment", "site"]]
    if sample.empty:
        parser.error("no participants coded female=0")
    if sample.isna().any().any():
        parser.error("chosen analysis columns contain missing values")
    if sample["treatment"].nunique() != 2:
        parser.error("chosen subgroup must contain both treatment assignments")

    result = smf.ols(
        "fin_security_m12 ~ treatment + C(site)", data=sample
    ).fit(cov_type="HC2", use_t=True)

    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
