#!/usr/bin/env python3
"""Score the prespecified sedentary-job year-2 treatment contrast.

Fits OLS with treatment, randomization-block indicators, and baseline sedentary
job status, using complete cases and HC2 robust standard errors.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


OUTCOME = "sedentary_job_y2"
BASELINE = "sedentary_job_base"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"treatment", "block", BASELINE, OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        parser.error("CSV is missing required columns: " + ", ".join(sorted(missing)))

    analysis = data[["treatment", "block", BASELINE, OUTCOME]].dropna()
    if analysis.empty:
        parser.error("No complete observations for the selected analysis")

    model = smf.ols(
        f"{OUTCOME} ~ treatment + C(block) + {BASELINE}", data=analysis
    ).fit(cov_type="HC2", use_t=True)

    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=model.df_resid))
    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
