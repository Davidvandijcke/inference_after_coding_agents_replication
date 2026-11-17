"""Re-run the single pre-selected Pathways to Work analysis on another dataset.

Chosen analysis (exploratory, selected from a larger search on the original data):
  outcome  : training_m6 (enrolled in education/job training at 6 months)
  subgroup : participants without college experience (education <= 2)
  model    : OLS of outcome on treatment + site indicators, HC2 robust SEs,
             two-sided t-test of the treatment coefficient (CODEBOOK convention).
No search is repeated; p_value is the unadjusted p-value for this one test.
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

d = pd.read_csv(a.data)
sub = d[d["education"] <= 2]
m = smf.ols("training_m6 ~ treatment + C(site)", data=sub).fit(cov_type="HC2", use_t=True)
# use_t with HC2 gives a t(df_resid) reference distribution
out = {
    "estimate": float(m.params["treatment"]),
    "std_error": float(m.bse["treatment"]),
    "p_value": float(m.pvalues["treatment"]),
}
with open(a.output, "w") as f:
    json.dump(out, f, indent=2)
