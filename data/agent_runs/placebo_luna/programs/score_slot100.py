#!/usr/bin/env python3
"""Score the treatment effect on requests for couple-finance information."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


OUTCOME = "wants_couple_finance_post"


def score(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    sample = data[[OUTCOME, "treatment", "block"]].dropna().copy()

    block_indicators = pd.get_dummies(
        sample["block"].astype("category"), drop_first=True, dtype=float
    )
    design = pd.concat([sample[["treatment"]].astype(float), block_indicators], axis=1)
    design = sm.add_constant(design)
    result = sm.OLS(sample[OUTCOME], design).fit(cov_type="HC2", use_t=True)

    return {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(score(args.data), indent=2) + "\n")


if __name__ == "__main__":
    main()
