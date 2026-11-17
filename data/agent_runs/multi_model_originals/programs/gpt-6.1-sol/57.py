#!/usr/bin/env python3
"""Locked test: full-sample intention-to-treat effect on employment at 6 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, statsmodels (and their dependencies).
Returns effects in probability units, not percentage points. No model selection,
subgroup selection, or multiplicity adjustment is performed by this program.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path, usecols=['site', 'treatment', 'employed_m6'])
    if data.empty or data.isna().any().any():
        raise ValueError('The selected analysis requires complete, nonempty data.')
    for column in ('treatment', 'employed_m6'):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be coded 0 or 1.')
    if data.treatment.nunique() != 2:
        raise ValueError('Both treatment arms are required.')
    design = pd.concat([
        pd.Series(1.0, index=data.index, name='intercept'),
        data[['treatment']].astype(float),
        pd.get_dummies(data.site.astype('category'), prefix='site',
                       drop_first=True, dtype=float),
    ], axis=1)
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Insufficient data to identify the site-adjusted treatment effect.')
    fit = sm.OLS(data.employed_m6.astype(float), design).fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The selected analysis produced nonfinite inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
