#!/usr/bin/env python3
"""Score the 12-month employment effect among participants unemployed at baseline."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    subgroup = data.loc[data["employed_baseline"] == 0]
    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=subgroup).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
