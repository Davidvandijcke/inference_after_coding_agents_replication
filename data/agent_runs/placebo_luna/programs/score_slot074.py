#!/usr/bin/env python3
"""Score the exploratory treatment-by-topic-good-use interaction.

Fits OLS of effect_per_unit on treatment, uses_topic_good, their interaction,
and randomization-block indicators with block-by-use-status terms. Reports
the treatment interaction with HC2 robust standard error and a two-sided
t-test, following CODEBOOK.md.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"effect_per_unit", "treatment", "uses_topic_good", "block"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(sorted(missing)))

    analysis = data[
        ["effect_per_unit", "treatment", "uses_topic_good", "block"]
    ].dropna()
    model = smf.ols(
        "effect_per_unit ~ treatment * uses_topic_good + C(block) * uses_topic_good",
        data=analysis,
    ).fit(cov_type="HC2", use_t=True)

    term = "treatment:uses_topic_good"
    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
