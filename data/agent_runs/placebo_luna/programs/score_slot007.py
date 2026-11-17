#!/usr/bin/env python3
"""Score the randomized evaluation's follow-up discussion outcome."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "discussed_anyone_fu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file with evaluation data")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    analysis = data[["treatment", "block", OUTCOME]].dropna()
    model = smf.ols(f'{OUTCOME} ~ treatment + C(block)', data=analysis)
    result = model.fit(cov_type="HC2")

    report = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
