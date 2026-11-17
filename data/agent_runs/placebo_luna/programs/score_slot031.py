#!/usr/bin/env python3
"""Estimate the block-adjusted treatment effect on immediate anger reports."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


OUTCOME = "felt_angry_post"


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {OUTCOME, "treatment", "block"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sample = data[[OUTCOME, "treatment", "block"]].dropna().copy()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("Need nonmissing outcome observations from both treatment arms")

    block_indicators = pd.get_dummies(
        sample["block"].astype("category"), prefix="block", drop_first=True, dtype=float
    )
    design = pd.concat([sample[["treatment"]].astype(float), block_indicators], axis=1)
    design = sm.add_constant(design, has_constant="add")

    fitted = sm.OLS(sample[OUTCOME].astype(float), design).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV dataset")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
