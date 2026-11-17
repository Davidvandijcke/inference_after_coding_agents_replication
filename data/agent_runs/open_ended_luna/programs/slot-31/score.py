#!/usr/bin/env python3
"""Score the 6-month employment effect for participants with some college."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the treatment effect on 6-month employment among "
            "participants with some college, adjusting for site."
        )
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"education", "employed_m6", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    subgroup = data.loc[data["education"] == 3]
    if subgroup.empty:
        parser.error("input CSV has no participants with education == 3")

    model = smf.ols(
        "employed_m6 ~ treatment + C(site)", data=subgroup
    ).fit()
    robust = model.get_robustcov_results(cov_type="HC2", use_t=True)
    treatment_index = model.model.exog_names.index("treatment")

    result = {
        "estimate": float(robust.params[treatment_index]),
        "std_error": float(robust.bse[treatment_index]),
        "p_value": float(robust.pvalues[treatment_index]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
