#!/usr/bin/env python3
"""Score the prespecified follow-up colleague-discussion contrast.

Fits the codebook's block-adjusted OLS model on observed outcomes and reports
the treatment coefficient, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t


OUTCOME = "discussed_colleagues_fu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = ["treatment", "block", OUTCOME]
    missing = [column for column in needed if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    analysis_data = data[needed].dropna()
    if analysis_data.empty:
        raise ValueError("No complete observations for the selected analysis")

    model = smf.ols(
        f"{OUTCOME} ~ treatment + C(block)", data=analysis_data
    ).fit(cov_type="HC2", use_t=True)
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    t_stat = estimate / std_error
    p_value = float(2 * student_t.sf(abs(t_stat), df=model.df_resid))

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "estimate": estimate,
                "std_error": std_error,
                "p_value": p_value,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
