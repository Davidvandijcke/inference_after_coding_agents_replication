#!/usr/bin/env python
"""Re-run the single chosen analysis (no search) on a dataset with the same columns.

Chosen analysis: effect of Pathways to Work on the 12-month financial security
index (fin_security_m12) among male participants (female == 0), estimated by OLS of
the outcome on treatment plus site indicators, HC2 standard errors, two-sided
t-test of the treatment coefficient (CODEBOOK.md convention).
"""
import argparse, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "fin_security_m12"
SUBGROUP_VAR, SUBGROUP_VAL = "female", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    d = d[d[SUBGROUP_VAR] == SUBGROUP_VAL]
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    res = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
