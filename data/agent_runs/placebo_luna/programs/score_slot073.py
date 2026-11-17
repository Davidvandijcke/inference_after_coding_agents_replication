#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on follow-up inability to cope."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

OUTCOME = "cannot_cope_fu"


def score(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"treatment", "block", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sample = data[["treatment", "block", OUTCOME]].dropna()
    block_indicators = pd.get_dummies(
        sample["block"].astype("category"), drop_first=True, dtype=float
    )
    design = pd.concat([sample[["treatment"]], block_indicators], axis=1)
    design = sm.add_constant(design, has_constant="add")
    result = sm.OLS(sample[OUTCOME], design).fit(cov_type="HC2", use_t=True)

    return {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the block-adjusted treatment effect on cannot_cope_fu."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON output")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
