#!/usr/bin/env python3
"""Score the one-year effect on physician/clinic/ER visits.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "physician_or_er_visit_y1"
BASELINE = "physician_or_er_visit_base"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {OUTCOME, BASELINE, "treatment", "block"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))

    analysis = data[[OUTCOME, BASELINE, "treatment", "block"]].dropna()
    if analysis.empty:
        parser.error("no complete observations for the selected analysis")

    fit = smf.ols(
        f"{OUTCOME} ~ treatment + C(block) + {BASELINE}", data=analysis
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
