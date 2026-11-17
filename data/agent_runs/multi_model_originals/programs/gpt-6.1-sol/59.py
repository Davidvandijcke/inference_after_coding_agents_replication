#!/usr/bin/env python3
"""Frozen Pathways to Work analysis: change in training enrollment effect.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels (with scipy).

Estimates OLS(training_m12 - training_m6 ~ treatment + site indicators),
with an intercept, HC2 standard errors, and a two-sided Student t test
using residual degrees of freedom. No outcome/subgroup/model search occurs.
The estimate is a probability difference, not percentage points.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ['site', 'treatment', 'training_m6', 'training_m12']
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    data = data[required].copy()
    if data.empty or data.isna().any().any():
        raise ValueError('Required columns must contain complete observations.')
    for name in ['treatment', 'training_m6', 'training_m12']:
        data[name] = pd.to_numeric(data[name], errors='raise')
        if not data[name].isin([0, 1]).all():
            raise ValueError(f'{name} must contain only 0 and 1.')
    if data.treatment.nunique() != 2:
        raise ValueError('Both treatment arms are required.')

    site_indicators = pd.get_dummies(
        data['site'], prefix='site', drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[['treatment']].astype(float), site_indicators], axis=1),
        has_constant='add',
    )
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('The regression requires full rank and positive residual degrees of freedom.')
    outcome = data['training_m12'].astype(float) - data['training_m6'].astype(float)
    model = sm.OLS(outcome, design).fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The regression did not produce finite inference statistics.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
