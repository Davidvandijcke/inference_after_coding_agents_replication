#!/usr/bin/env python3
"""Score the education interaction for 12-month financial security.

Fits the single site-adjusted OLS interaction selected in the accompanying
analysis. The reported estimate is the difference in the treatment effect
between participants with less than high school education and all others.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "education", "fin_security_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    data["low_education"] = (data["education"] == 1).astype(int)
    model = smf.ols(
        "fin_security_m12 ~ treatment * low_education + C(site)", data=data
    ).fit(cov_type="HC2")

    term = "treatment:low_education"
    estimate = float(model.params[term])
    std_error = float(model.bse[term])
    t_stat = estimate / std_error
    p_value = float(2 * student_t.sf(abs(t_stat), df=model.df_resid))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
