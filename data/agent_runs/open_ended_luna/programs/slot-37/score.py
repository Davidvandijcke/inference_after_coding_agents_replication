#!/usr/bin/env python3
"""Score the selected Pathways to Work analysis on a compatible CSV.

The analysis is the site-adjusted intention-to-treat effect on 6-month
employment among participants who were unemployed at baseline.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_baseline", "employed_m6"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    subgroup = data.loc[data["employed_baseline"].eq(0)]
    if subgroup.empty:
        raise ValueError("No participants were unemployed at baseline")

    model = smf.ols("employed_m6 ~ treatment + C(site)", data=subgroup).fit()
    robust = model.get_robustcov_results(cov_type="HC2", use_t=True)
    treatment_index = model.model.exog_names.index("treatment")

    return {
        "estimate": float(robust.params[treatment_index]),
        "std_error": float(robust.bse[treatment_index]),
        "p_value": float(robust.pvalues[treatment_index]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the selected site-adjusted Pathways to Work analysis."
    )
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
