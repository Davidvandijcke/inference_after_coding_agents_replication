#!/usr/bin/env python3
"""Score the preselected six-month earnings heterogeneity contrast.

The test compares the treatment effect at site 2 with the participant-count-
weighted mean treatment effect at the other observed sites. It fits one OLS
model with treatment-by-site interactions and uses HC2 robust covariance with
a two-sided t test. This script does not search outcomes or subgroups.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.stats import t


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "earnings_m6"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("site, treatment, and earnings_m6 must not contain missing values")

    sites = sorted(data["site"].unique().tolist())
    if 2 not in sites or len(sites) < 2:
        raise ValueError("The data must include site 2 and at least one other site")
    data = data.copy()
    data["site"] = pd.Categorical(data["site"], categories=sites, ordered=True)

    for site in sites:
        arms = set(data.loc[data["site"] == site, "treatment"].unique())
        if arms != {0, 1}:
            raise ValueError(f"Site {site} must contain both treatment and control participants")

    model = smf.ols("earnings_m6 ~ treatment * C(site)", data=data).fit()
    robust = model.get_robustcov_results(cov_type="HC2")
    design_info = model.model.data.design_info

    def site_effect_vector(site: int) -> np.ndarray:
        rows = pd.DataFrame(
            {
                "treatment": [1, 0],
                "site": pd.Categorical([site, site], categories=sites, ordered=True),
            }
        )
        matrix = np.asarray(
            build_design_matrices([design_info], rows, return_type="matrix")[0]
        )
        return matrix[0] - matrix[1]

    counts = data["site"].value_counts()
    other_sites = [site for site in sites if site != 2]
    other_total = sum(int(counts[site]) for site in other_sites)
    contrast = site_effect_vector(2)
    for site in other_sites:
        contrast -= (int(counts[site]) / other_total) * site_effect_vector(site)

    params = np.asarray(robust.params)
    covariance = np.asarray(robust.cov_params())
    estimate = float(contrast @ params)
    variance = float(contrast @ covariance @ contrast)
    std_error = float(np.sqrt(variance))
    t_statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(t_statistic), df=model.df_resid))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="CSV input path")
    parser.add_argument("--output", required=True, type=Path, help="JSON output path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
