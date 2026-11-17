#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on month-12 employment."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the Pathways to Work effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_m12", "treatment", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(missing))

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2"
    )
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    test_statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(test_statistic), df=model.df_resid))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
