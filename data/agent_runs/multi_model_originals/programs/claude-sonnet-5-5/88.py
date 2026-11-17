"""Re-run the single pre-selected analysis: effect of Pathways to Work on
employment at 6 months (employed_m6) among participants at site 2.

OLS of employed_m6 on treatment (site indicators are constant within a single
site, so only the intercept remains), HC2 robust standard errors, two-sided
t-test of the treatment coefficient. No other outcomes/subgroups are examined.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

SITE, OUTCOME = 2, "employed_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    s = d[d["site"] == SITE]
    if s["treatment"].nunique() < 2:
        raise SystemExit(f"site {SITE} needs both treated and control rows")
    X = sm.add_constant(s[["treatment"]].astype(float))
    r = sm.OLS(s[OUTCOME].astype(float), X).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(r.params["treatment"]),
        "std_error": float(r.bse["treatment"]),
        "p_value": float(r.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
