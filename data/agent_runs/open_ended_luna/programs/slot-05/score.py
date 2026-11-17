#!/usr/bin/env python3
"""Score the 12-month financial-security treatment-effect contrast by education.

The estimate is the treatment effect for participants with less than high
school education minus the treatment effect for participants with at least a
high school education, from an OLS model with site indicators. Inference uses
HC2 robust standard errors and a two-sided t-test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "education", "fin_security_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    data["low_education"] = (data["education"] == 1).astype(int)
    model = smf.ols(
        "fin_security_m12 ~ treatment * low_education + C(site) * low_education",
        data=data,
    ).fit()
    robust = model.get_robustcov_results(cov_type="HC2", use_t=True)
    coefficient = model.model.exog_names.index("treatment:low_education")

    result = {
        "estimate": float(robust.params[coefficient]),
        "std_error": float(robust.bse[coefficient]),
        "p_value": float(robust.pvalues[coefficient]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
