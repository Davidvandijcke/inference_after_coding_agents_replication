#!/usr/bin/env python3
"""Run the fixed, exploratory Pathways to Work analysis.

Population: baseline education 3 or 4 (some college or a degree).
Outcome: debt_collections_m6, in probability units.
Model: OLS on an intercept, treatment, and site indicators; HC2 standard
errors and a two-sided Student t test with residual degrees of freedom.

The outcome, follow-up, subgroup, and specification are fixed here. The
reported p-value is nominal; it does not account for the exploratory search
that selected this analysis. Requires pandas, numpy, scipy, and statsmodels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data: pd.DataFrame) -> dict:
    required = ["education", "site", "treatment", "debt_collections_m6"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns must be fully observed.")

    sample = data.loc[data["education"].isin([3, 4]), required].copy()
    if set(sample["treatment"].unique()) != {0, 1}:
        raise ValueError("The selected subgroup must contain both treatment arms.")
    if not sample["debt_collections_m6"].isin([0, 1]).all():
        raise ValueError("debt_collections_m6 must be binary (0 or 1).")

    design = pd.concat(
        [
            pd.Series(1.0, index=sample.index, name="intercept"),
            sample["treatment"].astype(float),
            pd.get_dummies(sample["site"], prefix="site", drop_first=True,
                           dtype=float),
        ],
        axis=1,
    )
    if len(sample) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The selected subgroup has an unidentified regression.")
    result = sm.OLS(sample["debt_collections_m6"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The selected analysis did not produce finite inference.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    output = chosen_analysis(pd.read_csv(args.data))
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
