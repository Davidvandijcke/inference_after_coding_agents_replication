"""Rerun the single chosen analysis: effect of Pathways to Work on training_m12
among female participants (female == 1), per the CODEBOOK convention:
OLS of outcome on treatment + site indicators, HC2 standard errors,
two-sided t-test on the treatment coefficient. No searching is done here.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json
import sys

sys.path.insert(0, "/opt/scientific-python")  # environment-specific; harmless elsewhere

import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    s = d[d["female"] == 1]
    m = smf.ols("training_m12 ~ treatment + C(site)", s).fit(cov_type="HC2")
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
