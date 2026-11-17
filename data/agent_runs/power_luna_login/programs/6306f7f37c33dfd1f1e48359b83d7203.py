#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    site_indicators = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat(
        [data[["treatment"]].astype(float), site_indicators], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    result = sm.OLS(data["employed_m12"].astype(float), design).fit(cov_type="HC2")

    estimate = float(result.params["treatment"])
    std_error = float(result.bse["treatment"])
    p_value = float(2 * t.sf(abs(estimate / std_error), result.df_resid))
    output = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
