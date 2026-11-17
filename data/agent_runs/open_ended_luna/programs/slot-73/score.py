#!/usr/bin/env python3
"""Score the selected exploratory treatment-effect heterogeneity analysis.

Fits the 6-month financial-security outcome on treatment, parent status, their
interaction, and site indicators. Reports the treatment-by-parent interaction
using HC2 robust standard errors and the conventional two-sided t-test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the participant CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    # Parent-specific site indicators make the interaction equal the difference
    # between the site-adjusted treatment effects estimated within each group.
    model = smf.ols(
        "fin_security_m6 ~ treatment * parent + C(site) * parent", data=data
    ).fit(cov_type="HC2")

    term = "treatment:parent"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
