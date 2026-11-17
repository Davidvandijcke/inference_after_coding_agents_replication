#!/usr/bin/env python3
"""Score the Democratic-versus-Independent treatment-effect contrast.

Fits OLS for capped effective-scenario willingness to pay, with treatment,
block indicators, political-affiliation indicators, and treatment-by-politics
interactions. Reports the contrast (treatment effect for Independents minus
treatment effect for Democrats), using HC2 standard errors and a two-sided
t-test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


OUTCOME = "wtp_effective_capped"
REQUIRED = {"block", "treatment", "politics", OUTCOME}


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    missing_columns = sorted(REQUIRED.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    data = data.dropna(subset=sorted(REQUIRED)).copy()
    if data.empty:
        raise ValueError("No complete observations for the selected analysis")
    if not set(data["politics"].unique()).issuperset({1, 2, 3}):
        raise ValueError("The analysis requires politics codes 1, 2, and 3")
    if not set(data["treatment"].unique()).issuperset({0, 1}):
        raise ValueError("The analysis requires treatment assignments 0 and 1")

    # Group 1 (Republican) is the reference. Block 1 (or the smallest observed
    # block code) is the reference block; block indicators adjust for the
    # randomization strata.
    design = pd.DataFrame(index=data.index)
    design["intercept"] = 1.0
    design["treatment"] = data["treatment"].astype(float)
    for block in sorted(data["block"].unique())[1:]:
        design[f"block_{block}"] = (data["block"] == block).astype(float)
    design["politics_2"] = (data["politics"] == 2).astype(float)
    design["politics_3"] = (data["politics"] == 3).astype(float)
    design["treatment_politics_2"] = design["treatment"] * design["politics_2"]
    design["treatment_politics_3"] = design["treatment"] * design["politics_3"]

    fitted = sm.OLS(data[OUTCOME].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    # Independent-minus-Democratic treatment effects:
    # (treatment + treatment_politics_3) -
    # (treatment + treatment_politics_2).
    contrast = np.zeros(len(fitted.params))
    contrast[fitted.params.index.get_loc("treatment_politics_3")] = 1.0
    contrast[fitted.params.index.get_loc("treatment_politics_2")] = -1.0
    result = fitted.t_test(contrast)
    return {
        "estimate": float(np.asarray(result.effect).item()),
        "std_error": float(np.asarray(result.sd).item()),
        "p_value": float(np.asarray(result.pvalue).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
