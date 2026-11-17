#!/usr/bin/env python3
"""Frozen analysis: 12-month distress, all randomized participants.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, statsmodels (and its scipy dependency).
OLS: distress_m12 ~ treatment + C(site), HC2 covariance, two-sided
Student t test with residual degrees of freedom. No search or model selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    columns = ['distress_m12', 'treatment', 'site']
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError('Analysis columns must be nonempty and fully observed.')
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors='raise')
    if not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError('Analysis columns must contain finite values.')
    if set(data.treatment.unique()) != {0, 1}:
        raise ValueError('Treatment must contain both 0 and 1, and no other values.')
    model = smf.ols('distress_m12 ~ treatment + C(site)', data=data,
                    missing='raise')
    design = model.exog
    if (len(data) <= design.shape[1]
            or np.linalg.matrix_rank(design) != design.shape[1]):
        raise ValueError('The model requires a full-rank design and residual degrees of freedom.')
    # HC2 is undefined when a participant has leverage one.
    q, _ = np.linalg.qr(design, mode='reduced')
    if np.any(1 - np.sum(q * q, axis=1) <= 1e-12):
        raise ValueError('HC2 requires leverage strictly below one.')
    result = model.fit(cov_type='HC2', use_t=True)
    answer = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in answer.values()):
        raise ValueError('Analysis produced a nonfinite result.')
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    answer = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(answer, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
