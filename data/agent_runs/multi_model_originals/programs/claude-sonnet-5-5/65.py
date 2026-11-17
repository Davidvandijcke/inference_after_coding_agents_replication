"""Rerun the single chosen analysis: effect of treatment on 12-month earnings
among participants under age 35. OLS with site indicators, HC2 SEs, two-sided t-test."""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

d = pd.read_csv(a.data)
s = d[d["age"] < 35]
m = smf.ols("earnings_m12 ~ treatment + C(site)", s).fit(cov_type="HC2")
out = {"estimate": float(m.params["treatment"]),
       "std_error": float(m.bse["treatment"]),
       "p_value": float(m.pvalues["treatment"])}
with open(a.output, "w") as f:
    json.dump(out, f)
