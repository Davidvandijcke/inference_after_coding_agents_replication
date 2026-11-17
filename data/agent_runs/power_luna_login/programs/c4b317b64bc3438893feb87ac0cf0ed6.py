#!/usr/bin/env python3
"""Score the prespecified lead analysis for Pathways to Work.

Fits 12-month employment on treatment and site fixed effects, with HC2 robust
standard errors. The treatment test is two-sided and uses a t reference
distribution, following the analysis convention in CODEBOOK.md.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    result = model.fit(cov_type="HC2", use_t=True)

    score = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(score, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
