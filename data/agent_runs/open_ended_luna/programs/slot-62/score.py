#!/usr/bin/env python3
"""Score the prespecified follow-up test for the Pathways to Work evaluation.

The test is the treatment-by-baseline-employment interaction for employment
at month 6, controlling for randomization site. The reported estimate is the
difference in treatment effects between participants employed at baseline
and those not employed at baseline.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_m6", "treatment", "employed_baseline", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input is missing required columns: " + ", ".join(sorted(missing)))

    fit = smf.ols(
        "employed_m6 ~ treatment * employed_baseline "
        "+ C(site) * employed_baseline",
        data=data,
    ).fit(cov_type="HC2", use_t=True)

    interaction = "treatment:employed_baseline"
    result = {
        "estimate": float(fit.params[interaction]),
        "std_error": float(fit.bse[interaction]),
        "p_value": float(fit.pvalues[interaction]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
