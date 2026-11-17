"""Re-run the chosen analysis: effect of Pathways to Work on training enrollment
at 6 months (training_m6), full sample, OLS on treatment + site indicators,
HC2 robust SE, two-sided t-test on the treatment coefficient (CODEBOOK convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    r = smf.ols("training_m6 ~ treatment + C(site)", d).fit(cov_type="HC2")
    out = {
        "estimate": float(r.params["treatment"]),
        "std_error": float(r.bse["treatment"]),
        "p_value": float(r.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
