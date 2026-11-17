#!/usr/bin/env python3
"""Reproduce selected analysis p_d40f62 only; no specification search."""
import argparse
import json
import os

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t
from statsmodels.stats.sandwich_covariance import cov_cluster

PATH_ID = "p_d40f62"
FORMULA = "unemployed ~ migstock + age + female + employed + C(year)"
QUESTION = "On the whole, do you think it should or should not be the government's responsibility to provide a decent standard of living for the unemployed?"
NULL = "H0: the OLS coefficient on migstock is zero in unemployed ~ migstock + age + female + employed + C(year)."
SCALE = "Estimate and standard error are points on a 0–100 rescaling of the 0–1 survey answer, for a +10 percentage-point difference in immigrant stock. Positive means more support for government responsibility."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Supplied respondent data CSV")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    columns = ["unemployed", "migstock", "age", "female", "employed", "year", "iso_country"]
    sample = df[columns].dropna().copy()
    fit = smf.ols(FORMULA, data=sample).fit()

    groups = sample["iso_country"]
    cluster_count = int(groups.nunique())
    covariance = cov_cluster(fit, groups, use_correction=True)
    exposure_index = fit.model.exog_names.index("migstock")
    coefficient = float(fit.params.iloc[exposure_index])
    coefficient_se = float(covariance[exposure_index, exposure_index] ** 0.5)
    degrees_freedom = cluster_count - 1
    statistic = coefficient / coefficient_se
    p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
    critical_value = float(t.ppf(0.975, degrees_freedom))
    lower = coefficient - critical_value * coefficient_se
    upper = coefficient + critical_value * coefficient_se

    # +10 stock percentage points, converted from a 0–1 answer to 0–100 points.
    transform = 1000.0
    output = {
        "path_id": PATH_ID,
        "question": QUESTION,
        "null": NULL,
        "model": FORMULA,
        "estimate": coefficient * transform,
        "std_error": coefficient_se * transform,
        "p_value": p_value,
        "ci95": [lower * transform, upper * transform],
        "estimate_unit": "points on a 0–100 rescaling of the survey answer per +10 percentage points of immigrant stock",
        "positive_direction": "more support for government responsibility",
        "raw_coefficient_per_1pp_stock": coefficient,
        "raw_std_error_per_1pp_stock": coefficient_se,
        "sample": {
            "respondents": int(len(sample)),
            "countries": cluster_count,
            "country_waves": int((sample["iso_country"].astype(str) + "_" + sample["year"].astype(str)).nunique()),
            "years": sorted(int(x) for x in sample["year"].unique()),
            "complete_case_variables": columns,
            "weighting": "unweighted",
        },
        "uncertainty": "Country-clustered sandwich covariance with finite-cluster correction; two-sided t reference with countries minus one degrees of freedom; nominal 95% confidence interval.",
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as result_file:
        json.dump(output, result_file, indent=2, ensure_ascii=False)
        result_file.write("\n")


if __name__ == "__main__":
    main()
