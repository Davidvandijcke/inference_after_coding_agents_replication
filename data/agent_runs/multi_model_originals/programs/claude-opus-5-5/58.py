"""Pathways to Work: rerun the single chosen analysis.

Effect of assignment to the program on enrollment in education/job training
at the 6-month follow-up (training_m6), among participants employed at random
assignment (employed_baseline == 1).

Model (CODEBOOK analysis convention): OLS of training_m6 on treatment and site
indicators, HC2 heteroskedasticity-robust standard errors, two-sided t-test of
the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="CSV with the codebook's columns")
    ap.add_argument("--output", required=True, help="path for JSON results")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    sub = df.loc[df["employed_baseline"] == 1, ["training_m6", "treatment", "site"]].dropna()

    fit = smf.ols("training_m6 ~ treatment + C(site)", data=sub).fit(cov_type="HC2")
    # Two-sided t-test on the treatment coefficient (t reference, residual df).
    test = fit.t_test("treatment = 0", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(test.pvalue),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
