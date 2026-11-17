#!/usr/bin/env python3
"""Fixed exploratory finding: six-month food insecurity, education >= 3.

Usage: python score.py --data data.csv --output result.json

OLS on an intercept, treatment, and site indicators within participants whose
baseline education is 3 or 4. HC2 standard error; two-sided Student t test with
residual degrees of freedom. No baseline covariates, outcome/subgroup search,
or multiple-testing correction. Estimates are in probability units.

Requires: numpy, pandas, scipy, statsmodels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ['education', 'treatment', 'site', 'food_insecure_m6']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data[required].isna().any().any():
        raise ValueError('Required columns must be completely observed.')
    if not data.education.isin([1, 2, 3, 4]).all():
        raise ValueError('education must use the codebook values 1 through 4.')
    if not data.treatment.isin([0, 1]).all():
        raise ValueError('treatment must be binary (0/1).')
    if not data.food_insecure_m6.isin([0, 1]).all():
        raise ValueError('food_insecure_m6 must be binary (0/1).')
    sub = data.loc[data.education.isin([3, 4])]
    if sub.treatment.nunique() != 2:
        raise ValueError('The selected subgroup must include both treatment arms.')
    design = pd.concat([
        sub[['treatment']].astype(float),
        pd.get_dummies(sub.site, prefix='site', drop_first=True, dtype=float),
    ], axis=1)
    design = sm.add_constant(design, has_constant='add')
    matrix = design.to_numpy()
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError('Treatment is not identified after controlling for site.')
    if len(sub) <= matrix.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom.')
    # A unit leverage makes HC2 undefined (e.g. a singleton site).
    leverage = np.einsum('ij,ji->i', matrix, np.linalg.pinv(matrix))
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('HC2 is undefined for observations with unit leverage.')
    result = sm.OLS(sub.food_insecure_m6.astype(float), design).fit(
        cov_type='HC2', use_t=True)
    output = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError('The selected analysis returned non-finite results.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    output = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
