"""Re-run the single chosen analysis: effect of Pathways to Work on training
enrollment at 12 months (training_m12) among participants at site 3.

Convention (CODEBOOK.md): OLS of the outcome on treatment plus site indicators
(constant within a single-site subgroup), HC2 standard errors, two-sided t-test
on the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

SITE = 3
OUTCOME = "training_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    sub = df[df["site"] == SITE]
    if sub["treatment"].nunique() < 2:
        raise SystemExit(f"site {SITE} lacks both treatment arms in {a.data}")
    # C(site) is collinear with the intercept inside one site, so it is absorbed there
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", sub).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
