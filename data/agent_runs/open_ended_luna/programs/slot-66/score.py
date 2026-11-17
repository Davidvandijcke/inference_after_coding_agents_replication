#!/usr/bin/env python3
"""Score the exploratory education subgroup contrast in the Pathways trial.

Requires pandas and statsmodels. The one reported test is the treatment by
low-education interaction for six-month financial security, with site-by-group
fixed effects and HC2 standard errors.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data["low_education"] = (data["education"] == 1).astype(int)
    data["site_education_group"] = (
        data["site"].astype(str) + "_" + data["low_education"].astype(str)
    )

    model = smf.ols(
        "fin_security_m6 ~ treatment + treatment:low_education + C(site_education_group)",
        data=data,
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment:low_education"]),
        "std_error": float(model.bse["treatment:low_education"]),
        "p_value": float(model.pvalues["treatment:low_education"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
