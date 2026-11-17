#!/usr/bin/env python3
"""Score the chosen exploratory subgroup contrast for Pathways to Work.

The test compares the treatment effect on usual weekly hours at month 12
between participants living with a child under 18 and participants who are not.
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
    required = ["hours_m12", "treatment", "parent", "site"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    # Site-by-parent indicators allow the site-adjusted treatment contrasts
    # to be estimated separately within each parent-status group.
    model = smf.ols(
        "hours_m12 ~ treatment * parent + C(site) * parent",
        data=data[required].dropna(),
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment:parent"]),
        "std_error": float(model.bse["treatment:parent"]),
        "p_value": float(model.pvalues["treatment:parent"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
