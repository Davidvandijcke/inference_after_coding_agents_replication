#!/usr/bin/env python3
"""Score the treatment effect among political independents.

Fits the codebook OLS specification for wtp_zero_if_ineffective, restricted to
participants coded as politically independent (politics == 3): outcome on
treatment and randomization block indicators, with HC2 standard errors.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    independent = data.loc[data["politics"] == 3]
    model = smf.ols(
        "wtp_zero_if_ineffective ~ treatment + C(block)", data=independent
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
