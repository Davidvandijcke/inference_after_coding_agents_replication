#!/usr/bin/env python3
"""Fixed analysis: six-month self-rated health among baseline parent == 0.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, scipy (see requirements.txt).

The reported estimate is the treatment coefficient from OLS with an intercept
and site indicators within this subgroup. The standard error is HC2; the
two-sided p value uses Student's t with n - rank(X) degrees of freedom.
There is no outcome, follow-up, subgroup, or specification search here.
The p value is nominal, without the exploratory report's search correction.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def analyze(data_path):
    observations = []
    required = {"parent", "site", "treatment", "health_m6"}
    with open(data_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            parent = float(row["parent"])
            if parent not in (0.0, 1.0):
                raise ValueError(f"Invalid parent value at CSV line {line}")
            if parent != 0.0:
                continue
            values = tuple(float(row[c]) for c in ("site", "treatment", "health_m6"))
            site, treatment, health = values
            if (not all(np.isfinite(v) for v in values)
                    or site not in (1, 2, 3, 4)
                    or treatment not in (0, 1)
                    or health not in (1, 2, 3, 4, 5)):
                raise ValueError(f"Invalid analysis value at CSV line {line}")
            observations.append(values)
    if not observations:
        raise ValueError("No participants with parent == 0")

    data = np.asarray(observations, dtype=float)
    sites, treatment, y = data.T
    columns = [np.ones(len(y)), treatment]
    columns.extend((sites == site).astype(float) for site in sorted(set(sites))[1:])
    x = np.column_stack(columns)
    rank = np.linalg.matrix_rank(x)
    df = len(y) - rank
    if rank != x.shape[1] or df <= 0:
        raise ValueError("Analysis requires a full-rank design and positive residual degrees of freedom")

    # The pseudoinverse is used only after checking full column rank.
    x_pinv = np.linalg.pinv(x)
    beta = x_pinv @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,ji->i", x, x_pinv)
    if np.any(1.0 - leverage <= np.finfo(float).eps * 10):
        raise ValueError("HC2 is undefined for observations with unit leverage")
    hc2_variance = residual**2 / (1.0 - leverage)
    variance = float(np.sum(x_pinv[1]**2 * hc2_variance))
    std_error = float(np.sqrt(variance))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("The treatment standard error must be finite and positive")
    estimate = float(beta[1])
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df=df))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path")
    try:
        result = analyze(args.data)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
