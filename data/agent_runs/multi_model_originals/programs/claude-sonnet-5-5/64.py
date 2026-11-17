"""Rerun the single chosen analysis (no search).

Effect of Pathways to Work on 6-month financial security index
(fin_security_m6) among participants who do NOT live with a child under 18
(parent == 0). OLS of outcome on treatment + site indicators, HC2 standard
errors, two-sided t-test on the treatment coefficient (CODEBOOK convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "fin_security_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    g = d[d["parent"] == 0]
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", g).fit(cov_type="HC2")
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
