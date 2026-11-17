#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

Fits the site-adjusted treatment effect on 6-month financial security among
participants who did not live with a child at baseline, using HC2 standard
errors and a two-sided t-test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"parent", "treatment", "site", "fin_security_m6"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(sorted(missing)))

    subgroup = data.loc[data["parent"] == 0]
    model = smf.ols("fin_security_m6 ~ treatment + C(site)", data=subgroup).fit(
        cov_type="HC2", use_t=True
    )

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
