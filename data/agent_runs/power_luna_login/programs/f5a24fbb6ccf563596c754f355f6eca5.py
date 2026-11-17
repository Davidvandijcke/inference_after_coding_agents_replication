#!/usr/bin/env python3
"""Report the treatment-by-gender contrast in 12-month financial security."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model = smf.ols(
        "fin_security_m12 ~ treatment * female + C(site)", data=data
    ).fit(cov_type="HC2")
    result = {
        "estimate": float(model.params["treatment:female"]),
        "std_error": float(model.bse["treatment:female"]),
        "p_value": float(model.pvalues["treatment:female"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
