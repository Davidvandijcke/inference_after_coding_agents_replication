#!/usr/bin/env python3
"""Estimate the fixed, baseline-adjusted effect on 12-month employment.

Dependencies: pandas, statsmodels.
The estimate and standard error are in probability units (1 = 100 percentage
points). Inference uses HC2 and a two-sided Student t test with OLS residual
degrees of freedom. Only this analysis is run.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model = smf.ols(
        "employed_m12 ~ treatment + C(site) + age + female + parent"
        " + C(education) + household_size + employed_baseline"
        " + earnings_baseline + months_unemployed_baseline",
        data=data,
        missing="raise",
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
