"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month
self-rated health among Site 1 participants.

OLS of health_m12 on treatment (site 1 only, so the site indicator is
constant and absorbed by the intercept), HC2 robust SE, two-sided t-test.
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
    d = d[d["site"] == 1]
    m = smf.ols("health_m12 ~ treatment", d).fit(cov_type="HC2")
    out = {"estimate": float(m.params["treatment"]),
           "std_error": float(m.bse["treatment"]),
           "p_value": float(m.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    main()
