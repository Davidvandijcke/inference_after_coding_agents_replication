#!/usr/bin/env python3
"""Estimate the frozen Pathways to Work exploratory contrast.

Outcome: distress_m12 - distress_m6 (Kessler-6 points).
Model: OLS on treatment, an intercept, and site indicators.
Inference: HC2 robust standard error; two-sided Student t test with
OLS residual degrees of freedom. The p-value is nominal, without a
multiple-comparison correction. No outcome or subgroup search is performed.

Dependencies: numpy, pandas, scipy (via statsmodels), statsmodels.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ['treatment', 'site', 'distress_m6', 'distress_m12']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if data[required].isna().any().any():
        raise ValueError('Required analysis columns must have no missing values.')
    numeric = data[['treatment', 'distress_m6', 'distress_m12']].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Treatment and distress values must be finite numbers.')
    if set(numeric.treatment.unique()) != {0.0, 1.0}:
        raise ValueError('Treatment must contain both 0 and 1, and no other values.')

    outcome = numeric.distress_m12 - numeric.distress_m6
    site_indicators = pd.get_dummies(
        data.site.astype('category'), prefix='site', drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([numeric[['treatment']], site_indicators], axis=1),
        has_constant='add',
    )
    if len(data) <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom.')
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError('Treatment and site model is not identifiable.')
    model = sm.OLS(outcome, design).fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('Analysis did not produce finite estimates and inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = analyze(args.data)
    with args.output.open('w', encoding='utf-8') as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write('\n')


if __name__ == '__main__':
    main()
