"""Rerun the chosen analysis: effect of Pathways to Work on 12-month
psychological distress (distress_m12), full sample.

OLS of distress_m12 on treatment + site indicators, HC2 standard errors,
two-sided t-test on the treatment coefficient (CODEBOOK.md convention).
No search is performed; the outcome and specification are fixed.
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    m = smf.ols("distress_m12 ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
