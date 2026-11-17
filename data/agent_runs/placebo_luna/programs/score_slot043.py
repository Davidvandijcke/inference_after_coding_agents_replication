#!/usr/bin/env python3
"""Score the prespecified exploratory signal from the Family and Work Planning trial.

Estimates the treatment effect on stressed_fu among participants in age_group 4,
adjusting for the randomization block and using an HC2 standard error.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    sample = data.loc[data["age_group"] == 4, ["treatment", "block", "stressed_fu"]]
    sample = sample.dropna(subset=["treatment", "block", "stressed_fu"])

    # In this prespecified subgroup age_group and block are both 4, so block
    # indicators are absorbed by the intercept. OLS on treatment is equivalent
    # to the codebook's treatment-plus-block model within this subgroup.
    design = sm.add_constant(sample[["treatment"]].astype(float), has_constant="add")
    result = sm.OLS(sample["stressed_fu"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
