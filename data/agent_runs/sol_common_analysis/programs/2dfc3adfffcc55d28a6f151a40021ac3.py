"""Estimate the intention-to-treat effect on employment at 12 months.

Dependencies: pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
The estimate and standard error are on the 0-to-1 probability scale.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["treatment", "site", "employed_m12"]
    if data[required].isna().any().any():
        raise ValueError("Treatment, site and employed_m12 must be observed.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must be coded 0 or 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment arms are required.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("Employed_m12 must be coded 0 or 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = sm.add_constant(pd.concat([data[["treatment"]], sites], axis=1),
                             has_constant="add")
    fit = sm.OLS(data["employed_m12"], design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
