#!/usr/bin/env python3
"""Score the selected exploratory contrast for a Pathways to Work dataset.

The analysis is the Site 4 treatment effect on 12-month food insecurity,
estimated by OLS with an intercept and treatment indicator and HC2 robust
standard errors. It deliberately runs this one analysis only.

Requires pandas and statsmodels.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    site4 = data.loc[data["site"] == 4]
    model = smf.ols("food_insecure_m12 ~ treatment", data=site4).fit(
        cov_type="HC2"
    )

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
