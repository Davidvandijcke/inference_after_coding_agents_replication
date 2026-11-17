#!/usr/bin/env python3
"""Score the year-one job-search finding among nonwhite participants."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "job_search_very_likely_y1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"white", "treatment", "block", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    analysis = data.loc[data["white"] == 0].dropna(subset=[OUTCOME])
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(block)", data=analysis).fit(
        cov_type="HC2", use_t=True
    )
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
