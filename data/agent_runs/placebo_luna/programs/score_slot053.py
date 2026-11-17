#!/usr/bin/env python3
"""Score the exploratory waste-topic treatment contrast on capped WTP."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"wtp_topic", "wtp_ineffective_capped", "treatment", "block"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # The selected subgroup is fixed in advance here: respondents assigned the
    # waste valuation topic. Block indicators preserve the randomization design.
    subset = data.loc[data["wtp_topic"].eq("waste")].copy()
    model = smf.ols(
        "wtp_ineffective_capped ~ treatment + C(block)", data=subset
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
