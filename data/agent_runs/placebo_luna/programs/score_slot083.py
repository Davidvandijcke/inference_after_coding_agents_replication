#!/usr/bin/env python3
"""Score the CO2-topic treatment effect on capped WTP if the reduction is ineffective."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data = data.loc[data["wtp_topic"] == "co2"].dropna(
        subset=["wtp_ineffective_capped", "treatment", "block"]
    )
    if data.empty:
        raise SystemExit("No complete observations in the CO2 valuation-topic group")

    model = smf.ols(
        'Q("wtp_ineffective_capped") ~ treatment + C(block)', data=data
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
