#!/usr/bin/env python3
"""Score the CO2-topic treatment effect on market-offset explanations."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with evaluation data")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"wtp_topic", "reason_market_offsets", "treatment", "block"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sample = data.loc[data["wtp_topic"].eq("co2"), [
        "reason_market_offsets", "treatment", "block"
    ]].dropna()
    if sample.empty:
        raise ValueError("No complete observations for wtp_topic == 'co2'")

    model = smf.ols(
        "reason_market_offsets ~ treatment + C(block)", data=sample
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
