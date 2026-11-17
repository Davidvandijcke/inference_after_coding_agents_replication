"""Pathways to Work: chosen exploratory finding, rerun on a new dataset.

Test: does the program effect on self-rated health at 12 months (health_m12)
vary with age at random assignment?

Model (CODEBOOK convention: OLS with site indicators, HC2 standard errors,
two-sided t-test):

    health_m12 ~ treatment + age10 + treatment:age10 + C(site)

where age10 = age / 10. The reported coefficient is treatment:age10, the
change in the treatment effect on health_m12 (1-5 scale) per 10 years of age.
The interaction coefficient does not change if age is shifted, so no
sample-specific centering is applied.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = "health_m12 ~ treatment * age10 + C(site)"
TERM = "treatment:age10"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    d = pd.read_csv(args.data, usecols=["site", "treatment", "age", "health_m12"])
    d = d.dropna()
    d["age10"] = d["age"] / 10.0

    fit = smf.ols(FORMULA, data=d).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params[TERM]),
        "std_error": float(fit.bse[TERM]),
        "p_value": float(fit.pvalues[TERM]),
        "test": "OLS of health_m12 on treatment x (age/10) with site indicators; "
                "HC2 SE; two-sided t-test of the treatment:age10 coefficient",
        "n": int(fit.nobs),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
