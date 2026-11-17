#!/usr/bin/env python3
"""Score the prespecified exploratory employment preference contrast."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


OUTCOMES = [
    "plan_employment_1y_post",
    "plan_employment_10y_post",
    "if_trusted_care_post",
    "if_no_norms_post",
    "if_higher_pay_post",
    "if_partner_flexible_post",
    "if_partner_engaged_post",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data = data.loc[data["age_group"] == 2].copy()
    data["employment_pref_composite_post"] = data[OUTCOMES].mean(axis=1)
    data = data.dropna(subset=["employment_pref_composite_post", "treatment", "block"])

    fit = smf.ols(
        "employment_pref_composite_post ~ treatment + C(block)", data=data
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
