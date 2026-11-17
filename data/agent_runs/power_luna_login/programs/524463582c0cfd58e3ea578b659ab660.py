#!/usr/bin/env python3
"""Score the 12-month employment effect in a Pathways to Work dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def estimate(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    outcome = "employed_m12"

    # Site indicators account for the within-site random assignment design.
    site_indicators = pd.get_dummies(
        data["site"].astype("category"),
        prefix="site",
        drop_first=True,
        dtype=float,
    )
    design = pd.concat([data[["treatment"]], site_indicators], axis=1)
    design = sm.add_constant(design, has_constant="add")

    # HC2 heteroskedasticity-robust SEs; use_t=True gives the codebook's
    # two-sided t test using the model residual degrees of freedom.
    model = sm.OLS(data[outcome], design).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
