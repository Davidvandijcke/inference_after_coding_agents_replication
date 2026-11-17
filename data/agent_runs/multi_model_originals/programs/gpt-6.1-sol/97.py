#!/usr/bin/env python3
"""Fixed analysis: six-month earnings ITT, adjusting only for site.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels, scipy.
The reported p-value is nominal, two-sided, and uses residual t degrees of
freedom. This program does not select an outcome or search across analyses.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    data = pd.read_csv(data_path)
    required = ['earnings_m6', 'treatment', 'site']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    data = data[required].copy()
    if data.empty or data.isna().any().any():
        raise ValueError('Analysis columns must contain complete observations.')
    for column in ['earnings_m6', 'treatment']:
        data[column] = pd.to_numeric(data[column], errors='raise')
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f'{column} must be finite.')
    if set(data['treatment'].unique()) != {0, 1}:
        raise ValueError('Treatment must contain both 0 and 1, and no other values.')

    model = smf.ols('earnings_m6 ~ treatment + C(site)', data=data)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Treatment and site effects are not separately identifiable.')
    if design.shape[0] <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The data do not permit finite HC2 inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
