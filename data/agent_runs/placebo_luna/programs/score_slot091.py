#!/usr/bin/env python3
"""Score the treatment effect on partner-engaged employment plans for ages 41–45."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "if_partner_engaged_post"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"age_group", "block", "treatment", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        parser.error("Dataset is missing required columns: " + ", ".join(sorted(missing)))

    analysis = data.loc[data["age_group"] == 4, [OUTCOME, "treatment", "block"]].dropna()
    if analysis.empty:
        parser.error("No complete observations for the selected analysis (age_group == 4)")

    result = smf.ols(f"{OUTCOME} ~ treatment + C(block)", data=analysis).fit(
        cov_type="HC2", use_t=True
    )
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
