#!/usr/bin/env python3
"""Score the prespecified lead analysis on a compatible evaluation dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "correct_ranking_post"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {OUTCOME, "treatment", "block", "works_full_time"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # The selected analysis is the randomized treatment contrast among baseline
    # full-time workers, adjusting for the randomization blocks.
    sample = data.loc[data["works_full_time"].eq(1), [OUTCOME, "treatment", "block"]]
    sample = sample.dropna(subset=[OUTCOME, "treatment", "block"])
    if sample.empty:
        raise ValueError("No complete observations in the full-time subgroup")

    result = smf.ols(f"{OUTCOME} ~ treatment + C(block)", data=sample).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
