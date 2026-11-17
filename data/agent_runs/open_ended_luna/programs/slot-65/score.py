#!/usr/bin/env python3
"""Score the six-month employment effect contrast between sites 2 and 4."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to an evaluation CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_baseline", "employed_m6"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not {2, 4}.issubset(set(data["site"].dropna().unique())):
        raise ValueError("The data must include participants from sites 2 and 4")

    # Site-specific treatment effects, adjusted for baseline employment.
    model = smf.ols(
        "employed_m6 ~ treatment * C(site) + employed_baseline", data=data
    ).fit(cov_type="HC2", use_t=True)

    # Contrast: treatment effect at site 2 minus treatment effect at site 4.
    names = list(model.params.index)
    contrast = np.zeros((1, len(names)))
    contrast[0, names.index("treatment:C(site)[T.2]")] = 1.0
    contrast[0, names.index("treatment:C(site)[T.4]")] = -1.0
    result = model.t_test(contrast)

    payload = {
        "estimate": float(np.asarray(result.effect).item()),
        "std_error": float(np.asarray(result.sd).item()),
        "p_value": float(np.asarray(result.pvalue).item()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
