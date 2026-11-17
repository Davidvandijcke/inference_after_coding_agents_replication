#!/usr/bin/env python3
"""Score the selected Pathways to Work analysis on a compatible dataset.

Chosen test: the program's effect on 12-month Kessler-6 distress among
participants with a bachelor's degree or higher, estimated by OLS with site
indicators and HC2 standard errors.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    if data_path.resolve() == output_path.resolve():
        parser.error("--output must not overwrite the input data")

    data = pd.read_csv(data_path)
    subgroup = data.loc[data["education"] == 4]
    if subgroup.empty or subgroup["treatment"].nunique() < 2:
        parser.error("The education == 4 subgroup must include both treatment arms")

    result = smf.ols(
        "distress_m12 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "estimate": float(result.params["treatment"]),
                "std_error": float(result.bse["treatment"]),
                "p_value": float(result.pvalues["treatment"]),
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
