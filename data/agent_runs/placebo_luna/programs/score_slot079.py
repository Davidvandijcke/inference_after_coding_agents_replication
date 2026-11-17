#!/usr/bin/env python3
"""Score the fixed full-time subgroup analysis for the Family and Work Planning trial.

Estimates the treatment effect on rank_childcare_post among participants who
worked full time at baseline, adjusting for randomization block and using HC2
robust standard errors. The analysis does not search across outcomes/groups.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"treatment", "block", "works_full_time", "rank_childcare_post"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(sorted(missing)))

    analysis = data.loc[
        data["works_full_time"].eq(1),
        ["treatment", "block", "rank_childcare_post"],
    ].dropna()
    if analysis["treatment"].nunique() != 2:
        parser.error("full-time subgroup must contain both treatment and control observations")

    block_indicators = pd.get_dummies(
        analysis["block"].astype("category"), drop_first=True, dtype=float
    ).reset_index(drop=True)
    design = pd.concat(
        [
            pd.Series(1.0, index=range(len(analysis)), name="intercept"),
            analysis["treatment"].astype(float).reset_index(drop=True).rename("treatment"),
            block_indicators,
        ],
        axis=1,
    )
    result = sm.OLS(
        analysis["rank_childcare_post"].astype(float).to_numpy(), design
    ).fit(cov_type="HC2")

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
