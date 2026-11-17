#!/usr/bin/env python
"""Re-run the single pre-chosen analysis on a dataset with the same columns.

Analysis: effect of Pathways to Work on debt in collections at 12 months
(debt_collections_m12), full sample. OLS of outcome on treatment + site
indicators, HC2 standard errors, two-sided t-test on the treatment coefficient
(CODEBOOK.md convention). No search is performed.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "debt_collections_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
