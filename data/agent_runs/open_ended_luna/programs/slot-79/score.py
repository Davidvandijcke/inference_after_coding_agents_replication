#!/usr/bin/env python3
"""Score the selected Pathways to Work treatment-by-household-size analysis.

Fits one site-adjusted OLS model for 6-month financial security and returns
the treatment-by-baseline-household-size coefficient with an HC2 standard
error and two-sided t-test p-value. This script does not search outcomes or
subgroups and does not apply a multiplicity adjustment.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "household_size", "fin_security_m6"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    # Centering makes the treatment main effect refer to the sample's mean
    # household size; it does not change the interaction estimate.
    data["household_size_centered"] = (
        data["household_size"] - data["household_size"].mean()
    )
    fit = smf.ols(
        "fin_security_m6 ~ treatment * household_size_centered + C(site)",
        data=data,
    ).fit(cov_type="HC2", use_t=True)

    term = "treatment:household_size_centered"
    result = {
        "estimate": float(fit.params[term]),
        "std_error": float(fit.bse[term]),
        "p_value": float(fit.pvalues[term]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
