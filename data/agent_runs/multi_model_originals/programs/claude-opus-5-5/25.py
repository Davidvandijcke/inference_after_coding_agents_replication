"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (fixed; no search is performed here):
  Outcome:  training_m6 (enrolled in education/job training at the 6-month survey)
  Sample:   participants aged under 35 at random assignment (age < 35)
  Model:    OLS of training_m6 on treatment + site indicators (CODEBOOK convention)
  Inference: HC2 robust standard error; two-sided t-test of the treatment coefficient
             (t reference distribution with residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "training_m6"
AGE_CUTOFF = 35


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    ap.add_argument("--output", required=True, help="path for the JSON result")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    sub = df.loc[df["age"] < AGE_CUTOFF, [OUTCOME, "treatment", "site"]].dropna()
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=sub).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
