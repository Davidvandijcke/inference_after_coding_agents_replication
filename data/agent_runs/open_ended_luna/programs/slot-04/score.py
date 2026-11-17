#!/usr/bin/env python3
"""Score the selected Pathways to Work education-group contrast.

Estimates how the treatment effect on 6-month self-rated health differs between
participants with high school education or less and those with some college or
more. The model allows site baselines to differ by education group and uses HC2
robust standard errors with a two-sided t test.

Requires pandas and statsmodels.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data["low_education"] = (data["education"] <= 2).astype(int)
    result = smf.ols(
        "health_m6 ~ treatment * low_education + C(site) * low_education",
        data=data,
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment:low_education"]),
        "std_error": float(result.bse["treatment:low_education"]),
        "p_value": float(result.pvalues["treatment:low_education"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
