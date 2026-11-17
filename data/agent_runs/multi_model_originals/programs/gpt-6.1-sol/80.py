#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis; no outcome/subgroup search.

Chosen estimand: intention-to-treat effect on six-month debt collections
among participants living with a child under 18 at baseline (parent == 1).
Model: debt_collections_m6 ~ treatment + C(site), with an intercept.
Inference: HC2 standard error and a two-sided Student t test using n - rank(X)
degrees of freedom. The estimate and standard error are in probability units,
so multiply them by 100 to express percentage points. The p-value is nominal;
it does not adjust for the exploratory search on the original dataset.

Requires numpy, pandas, and statsmodels (plus their dependencies).
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    required = ["parent", "treatment", "site", "debt_collections_m6"]
    data = pd.read_csv(data_path, usecols=required)
    if data[required].isna().any().any():
        raise ValueError("The selected analysis requires complete observations.")
    for column in ["parent", "treatment", "debt_collections_m6"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    parents = data.loc[data["parent"].eq(1)].copy()
    if parents.empty or parents["treatment"].nunique() != 2:
        raise ValueError("The parent subgroup must include both treatment arms.")

    model = smf.ols(
        "debt_collections_m6 ~ treatment + C(site)",
        data=parents,
        missing="raise",
    )
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The selected regression design is not full rank.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("The selected regression requires residual degrees of freedom.")
    if np.any(model.fit().get_influence().hat_matrix_diag >= 1 - 1e-12):
        raise ValueError("HC2 is undefined when an observation has unit leverage.")

    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The selected analysis did not produce finite inference.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data.")
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
