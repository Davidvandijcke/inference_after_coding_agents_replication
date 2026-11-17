"""Re-run the single chosen analysis: effect of Pathways to Work on food insecurity
at 6 months (food_insecure_m6), full sample, OLS on treatment + site indicators,
HC2 standard errors, two-sided t-test on the treatment coefficient (CODEBOOK convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "food_insecure_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    sites = pd.get_dummies(d["site"], prefix="site", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[["treatment"]].astype(float), sites], axis=1))
    r = sm.OLS(d[OUTCOME].astype(float), X).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(r.params["treatment"]),
        "std_error": float(r.bse["treatment"]),
        "p_value": float(r.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
