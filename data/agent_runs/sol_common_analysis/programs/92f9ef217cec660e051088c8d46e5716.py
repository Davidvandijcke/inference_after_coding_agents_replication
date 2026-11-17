#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels (which requires scipy).
The estimate and standard error are in probability units, not percentage points.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=['site', 'treatment', 'employed_m12'])
    if data.empty or data.isna().any().any():
        raise ValueError('The chosen analysis requires nonmissing observations.')
    for column in ['treatment', 'employed_m12']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be coded 0 or 1.')
    if data['treatment'].nunique() != 2:
        raise ValueError('Both randomized groups are required.')

    sites = pd.get_dummies(data['site'], prefix='site', drop_first=True,
                           dtype=float)
    design = sm.add_constant(pd.concat([data[['treatment']], sites], axis=1),
                             has_constant='add').astype(float)
    if (len(data) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]):
        raise ValueError('The regression requires a full-rank design and residual degrees of freedom.')

    result = sm.OLS(data['employed_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True)
    output = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    with args.output.open('w', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
