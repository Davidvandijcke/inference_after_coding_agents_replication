#!/usr/bin/env python3
"""Score the prespecified lead analysis: site-adjusted 12-month employment."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = {"employed_m12", "treatment", "site"}
    missing = needed.difference(data.columns)
    if missing:
        parser.error("missing required columns: " + ", ".join(sorted(missing)))

    site_indicators = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data["treatment"].astype(float).rename("treatment"),
            site_indicators,
        ],
        axis=1,
    )
    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
