#!/usr/bin/env python3
"""Score the selected exploratory sex interaction.

Fits site-adjusted OLS for debt collections at month 6, with a treatment by
female interaction, and reports the interaction coefficient (female minus male
treatment effect) with HC2 robust inference.
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
    fit = smf.ols(
        "debt_collections_m6 ~ treatment * female + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:female"
    result = {
        "estimate": float(fit.params[term]),
        "std_error": float(fit.bse[term]),
        "p_value": float(fit.pvalues[term]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
