#!/usr/bin/env python3
"""Score the selected exploratory sex-by-treatment employment contrast."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the treatment-by-female interaction for 12-month "
            "employment, adjusting for site with HC2 standard errors."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["employed_m12", "treatment", "female", "site"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = data.dropna(subset=required).copy()
    if data.empty:
        raise ValueError("No complete observations for the selected analysis")

    data["tx_female"] = data["treatment"] * data["female"]
    site = pd.get_dummies(
        data["site"].astype("category"), prefix="site", drop_first=True, dtype=float
    )
    # Let site-specific outcome levels differ by sex, as in separate
    # site-adjusted regressions within the two sex groups.
    site_female = site.mul(data["female"], axis=0).add_prefix("female_")
    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["treatment", "female", "tx_female"]].astype(float),
            site,
            site_female,
        ],
        axis=1,
    )

    fitted = sm.OLS(data["employed_m12"].astype(float), design).fit()
    robust = fitted.get_robustcov_results(cov_type="HC2", use_t=True)
    interaction_index = design.columns.get_loc("tx_female")
    result = {
        "estimate": float(robust.params[interaction_index]),
        "std_error": float(robust.bse[interaction_index]),
        "p_value": float(robust.pvalues[interaction_index]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
