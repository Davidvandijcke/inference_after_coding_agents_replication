"""Re-estimate the single pre-chosen analysis: effect of Pathways to Work on
earnings at 6 months (earnings_m6) among participants at site 3.

Convention (CODEBOOK.md): OLS of the outcome on treatment and site indicators,
HC2 standard errors, two-sided t-test of the treatment coefficient. Within a
single site the site indicators are constant and absorbed by the intercept.
No search over outcomes, follow-ups or subgroups is performed.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

SITE, OUTCOME = 3, "earnings_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    sub = d[d["site"] == SITE]
    m = smf.ols(f"{OUTCOME} ~ treatment", sub).fit(cov_type="HC2", use_t=True)
    res = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
