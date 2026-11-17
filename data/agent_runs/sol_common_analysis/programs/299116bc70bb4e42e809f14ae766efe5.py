"""Estimate the site-adjusted assignment effect on 12-month distress.

Requires pandas and statsmodels. Runs only this fixed analysis, with HC2
standard errors and a two-sided Student t test using residual degrees of freedom.
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
    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    )
    fit = sm.OLS(data["distress_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
