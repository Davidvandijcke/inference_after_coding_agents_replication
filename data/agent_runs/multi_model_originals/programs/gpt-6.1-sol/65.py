#!/usr/bin/env python3
"""Fixed exploratory finding: 12-month earnings ITT among adults under 35.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy.

OLS: earnings_m12 ~ intercept + treatment + categorical site indicators,
restricted to age < 35 at baseline. HC2 standard error; two-sided Student t
test with residual degrees of freedom. No baseline covariates, search, or
multiplicity adjustment. The p-value is nominal, conditional on this fixed test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def score(data_path):
    data = pd.read_csv(data_path)
    required = ['age', 'site', 'treatment', 'earnings_m12']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if data[required].isna().any().any():
        raise ValueError('Required columns must be fully observed.')
    if not np.isfinite(data[['age', 'treatment', 'earnings_m12']].to_numpy(dtype=float)).all():
        raise ValueError('Age, treatment, and earnings must be finite numeric values.')
    if not data.treatment.isin([0, 1]).all():
        raise ValueError('Treatment must be coded 0 or 1.')
    sub = data.loc[data.age < 35]
    if sub.treatment.nunique() != 2:
        raise ValueError('The under-35 subgroup must contain both assignment arms.')
    sites = pd.get_dummies(sub.site, drop_first=True, dtype=float).to_numpy()
    x = np.column_stack([np.ones(len(sub)), sub.treatment.to_numpy(dtype=float), sites])
    y = sub.earnings_m12.to_numpy(dtype=float)
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) != k:
        raise ValueError('The subgroup regression requires full rank and positive residual degrees of freedom.')
    # QR avoids forming the normal equations. bread = (X'X)^(-1).
    q, r = np.linalg.qr(x, mode='reduced')
    beta = np.linalg.solve(r, q.T @ y)
    residual = y - x @ beta
    leverage = np.sum(q * q, axis=1)
    if np.any(1 - leverage <= np.finfo(float).eps * 10):
        raise ValueError('HC2 is undefined for an observation with unit leverage.')
    # Row 1 of R^-1 Q' maps the observations to the treatment coefficient.
    influence = np.linalg.solve(r, q.T)[1]
    variance = np.sum(influence ** 2 * residual ** 2 / (1 - leverage))
    std_error = float(np.sqrt(variance))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError('The treatment standard error must be positive and finite.')
    estimate = float(beta[1])
    p_value = float(2 * t.sf(abs(estimate / std_error), df=n - k))
    return {'estimate': estimate, 'std_error': std_error, 'p_value': p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
