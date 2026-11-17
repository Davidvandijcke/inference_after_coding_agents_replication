#!/usr/bin/env python3
"""Score the fixed Pathways to Work analysis.

Chosen estimand: intention-to-treat effect on training_m12 in the full sample,
in probability units. OLS includes an intercept, treatment, and site indicators
(one reference site). Inference uses HC2 and a two-sided Student t test with
n - rank(X) residual degrees of freedom. No baseline covariates or search.

Dependencies: pandas, numpy, statsmodels, scipy.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    df = pd.read_csv(data_path, usecols=['training_m12', 'treatment', 'site'])
    if df.empty or df.isna().any().any():
        raise ValueError('The chosen analysis requires complete, nonempty data.')
    if not df['treatment'].isin([0, 1]).all() or df['treatment'].nunique() != 2:
        raise ValueError('Treatment must contain both 0 and 1, and no other values.')
    if not df['training_m12'].isin([0, 1]).all():
        raise ValueError('training_m12 must be coded 0 or 1.')

    X = pd.concat([
        df[['treatment']].astype(float),
        pd.get_dummies(df['site'], prefix='site', drop_first=True, dtype=float),
    ], axis=1)
    X = sm.add_constant(X, has_constant='add')
    matrix = X.to_numpy(dtype=float)
    if len(df) <= X.shape[1] or np.linalg.matrix_rank(matrix) != X.shape[1]:
        raise ValueError('The treatment/site model must have full rank and positive residual degrees of freedom.')
    q, _ = np.linalg.qr(matrix, mode='reduced')
    if np.any(1 - np.sum(q * q, axis=1) <= np.finfo(float).eps * 10):
        raise ValueError('HC2 is undefined for observations with unit leverage.')

    fit = sm.OLS(df['training_m12'].astype(float), X).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen test did not produce finite results.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
