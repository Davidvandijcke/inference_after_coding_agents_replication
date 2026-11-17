#!/usr/bin/env python3
"""Score the preselected exploratory subgroup analysis.

Estimates the treatment effect on market-offset explanations among respondents
aged 35–49, controlling for randomization block, with HC2 robust standard
errors and a two-sided t test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["age"].between(35, 49)].dropna(
        subset=["reason_market_offsets", "treatment", "block"]
    )
    result = smf.ols(
        "reason_market_offsets ~ treatment + C(block)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
