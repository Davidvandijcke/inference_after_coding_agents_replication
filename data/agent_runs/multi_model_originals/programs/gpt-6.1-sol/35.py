#!/usr/bin/env python3
"""Fixed analysis: assignment effect on debt_collections_m12 within site 3.

Run: python score.py --data data.csv --output result.json
Requires scipy. Only the supplied CSV is read; there is no model selection.

Within one site, OLS on an intercept and treatment is a difference in means.
Its HC2 variance is s_treatment**2/n_treatment + s_control**2/n_control.
Inference uses a two-sided Student t test with OLS residual df = n - 2,
not Welch-Satterthwaite degrees of freedom. Estimates are in probability units.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, variance

from scipy.stats import t


def analyze(data_path):
    arms = {0: [], 1: []}
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "debt_collections_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        for row_number, row in enumerate(reader, start=2):
            site = float(row["site"])
            if not math.isfinite(site):
                raise ValueError(f"Nonfinite site at CSV row {row_number}")
            if site != 3:
                continue
            treatment = float(row["treatment"])
            outcome = float(row["debt_collections_m12"])
            if treatment not in (0, 1) or outcome not in (0, 1):
                raise ValueError(
                    f"Treatment and outcome must be 0 or 1 at CSV row {row_number}"
                )
            arms[int(treatment)].append(outcome)

    control, treated = arms[0], arms[1]
    if min(len(control), len(treated)) < 2:
        raise ValueError("Site 3 must contain at least two observations in each arm")
    estimate = mean(treated) - mean(control)
    std_error = math.sqrt(
        variance(treated) / len(treated) + variance(control) / len(control)
    )
    if std_error == 0:
        raise ValueError("Zero HC2 standard error: the specified t test is undefined")
    degrees_of_freedom = len(treated) + len(control) - 2
    p_value = float(2 * t.sf(abs(estimate / std_error), degrees_of_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Refuse an accidental overwrite of the input observations.
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")


if __name__ == "__main__":
    main()
