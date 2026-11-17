#!/usr/bin/env python3
"""Score the 12-month employment treatment-effect difference by sex."""

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
    required = {"site", "treatment", "female", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # female is the reference group; the interaction coefficient is the
    # male-minus-female difference in treatment effects.
    male = 1 - data["female"]
    design = pd.DataFrame(
        {
            "treatment": data["treatment"],
            "male": male,
            "treatment_x_male": data["treatment"] * male,
        },
        index=data.index,
    )
    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(pd.concat([design, site_indicators], axis=1))

    result = sm.OLS(data["employed_m12"], design).fit(
        cov_type="HC2", use_t=True
    )
    payload = {
        "estimate": float(result.params["treatment_x_male"]),
        "std_error": float(result.bse["treatment_x_male"]),
        "p_value": float(result.pvalues["treatment_x_male"]),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
