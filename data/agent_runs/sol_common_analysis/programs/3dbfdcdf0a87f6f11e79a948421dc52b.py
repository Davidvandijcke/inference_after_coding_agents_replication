#!/usr/bin/env python3
"""Estimate the site-adjusted intention-to-treat effect on 12-month employment.

Requires pandas, numpy, and statsmodels. The outcome is a proportion, so the
estimate and standard error are in probability units. Inference uses HC2
standard errors and a two-sided t-test with OLS residual degrees of freedom.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    columns = ["employed_m12", "treatment", "site"]
    data = pd.read_csv(args.data, usecols=columns)
    if data.isna().any().any():
        raise ValueError("Analysis variables must be fully observed.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("treatment must contain both 0 and 1, and no other values.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    if (np.linalg.matrix_rank(model.exog) != model.exog.shape[1]
            or model.exog.shape[0] <= model.exog.shape[1]
            or np.any(np.sum(np.linalg.qr(model.exog, mode="reduced")[0] ** 2,
                             axis=1) >= 1 - 1e-12)):
        raise ValueError("Design must have full rank, residual degrees of freedom, "
                         "and leverages below one for HC2 inference.")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The requested analysis produced nonfinite results.")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
