"""Re-run the single chosen analysis on another dataset with the same columns.

Chosen analysis: heterogeneity of the program effect on 6-month financial
security (fin_security_m6) by household size. OLS of fin_security_m6 on
treatment, household_size, treatment x household_size and site indicators;
HC2 standard errors; two-sided t-test of the treatment x household_size
coefficient (change in treatment effect per additional household member).
No searching is done here.
"""
import argparse
import json
import sys

sys.path.append("/opt/scientific-python")
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    fit = smf.ols("fin_security_m6 ~ treatment * household_size + C(site)", d).fit(cov_type="HC2")
    term = "treatment:household_size"
    out = {
        "estimate": float(fit.params[term]),
        "std_error": float(fit.bse[term]),
        "p_value": float(fit.pvalues[term]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
