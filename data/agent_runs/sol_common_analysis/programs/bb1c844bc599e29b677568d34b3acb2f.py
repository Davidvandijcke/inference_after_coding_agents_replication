"""Estimate the assignment effect on six-month food insecurity.

Requires pandas, statsmodels, numpy, and scipy. Runs only the selected
full-sample regression: food_insecure_m6 ~ treatment + categorical site.
The returned p-value is the nominal, two-sided HC2 t-test p-value.
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

    data = pd.read_csv(args.data, usecols=["site", "treatment", "food_insecure_m6"])
    if data.isna().any().any():
        raise ValueError("The selected analysis requires complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["food_insecure_m6"].isin([0, 1]).all():
        raise ValueError("food_insecure_m6 must be binary (0 or 1).")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([data[["treatment"]], sites], axis=1))
    fit = sm.OLS(data["food_insecure_m6"], design.astype(float)).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
