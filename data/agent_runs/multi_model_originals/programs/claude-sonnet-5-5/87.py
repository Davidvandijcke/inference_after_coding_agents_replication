"""Re-estimate the single pre-specified candidate test from the exploratory analysis:
effect of Pathways to Work on 6-month psychological distress (distress_m6) in site 2.
Convention (CODEBOOK.md): OLS of outcome on treatment (site indicators are
constant within a single site, so only an intercept is added), HC2 robust SEs,
two-sided t-test of the treatment coefficient. No search is performed."""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

SITE, OUTCOME = 2, "distress_m6"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    d = d[d["site"] == SITE]
    m = smf.ols(f"{OUTCOME} ~ treatment", d).fit(cov_type="HC2", use_t=True)
    out = {"estimate": float(m.params["treatment"]),
           "std_error": float(m.bse["treatment"]),
           "p_value": float(m.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
