#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the site-adjusted program effect on 12-month financial "
            "security among participants with a high school diploma or GED."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"education", "site", "treatment", "fin_security_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    subgroup = data.loc[data["education"] == 2]
    if subgroup.empty or subgroup["treatment"].nunique() != 2:
        raise ValueError(
            "The education=2 subgroup must contain both treatment and control participants"
        )

    model = smf.ols(
        "fin_security_m12 ~ treatment + C(site)", data=subgroup
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
