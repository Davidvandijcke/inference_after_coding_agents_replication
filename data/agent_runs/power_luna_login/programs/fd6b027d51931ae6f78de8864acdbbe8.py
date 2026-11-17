#!/usr/bin/env python3
"""Score the preselected Pathways to Work subgroup employment analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with evaluation data")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"education", "employed_m6", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    # Preselected test: among participants with education <= high school/GED,
    # estimate the 6-month employment effect, adjusting for randomization site.
    subgroup = data.loc[data["education"] <= 2].copy()
    result = smf.ols("employed_m6 ~ treatment + C(site)", data=subgroup).fit(
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
