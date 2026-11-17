"""Pathways to Work: effect of program assignment on employment at 12 months.

Reruns the single chosen analysis (no search): OLS of employed_m12 on treatment
and site indicators, HC2 robust standard errors, two-sided t-test on treatment.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    ap.add_argument("--output", required=True, help="path for the JSON result")
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    d = d[["employed_m12", "treatment", "site"]].dropna()
    formula = "employed_m12 ~ treatment"
    if d["site"].nunique() > 1:
        formula += " + C(site)"
    fit = smf.ols(formula, data=d).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise SystemExit("non-finite result; check the treatment variable has variation")
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
