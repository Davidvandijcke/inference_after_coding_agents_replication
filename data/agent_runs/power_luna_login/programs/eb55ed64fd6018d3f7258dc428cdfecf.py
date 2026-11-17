#!/usr/bin/env python3
"""Score the prespecified exploratory lead: treatment effect heterogeneity by parent status on 12-month health."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"health_m12", "treatment", "parent", "site"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    # The interaction coefficient is the difference in the treatment effect
    # between participants living with children and those not living with children.
    fit = smf.ols("health_m12 ~ treatment * parent + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {
                "estimate": float(fit.params["treatment:parent"]),
                "std_error": float(fit.bse["treatment:parent"]),
                "p_value": float(fit.pvalues["treatment:parent"]),
            },
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
