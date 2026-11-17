#!/usr/bin/env python3
"""Run the fixed, exploratory Pathways to Work analysis.

Estimand: assignment effect on food_insecure_m12 within parent == 1.
Model: OLS with an intercept, treatment, and categorical site indicators.
Inference: HC2 standard error; two-sided t test with n - rank(X) degrees
of freedom. There is no outcome, subgroup, or model selection in this file.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, statsmodels (and their dependencies).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ['parent', 'treatment', 'site', 'food_insecure_m12']
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    values = data[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Required columns must contain finite, observed values.')
    for column in ('parent', 'treatment', 'food_insecure_m12'):
        if not values[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1.')

    sample = values.loc[values['parent'].eq(1)].reset_index(drop=True)
    if sample['treatment'].nunique() != 2:
        raise ValueError('The parent subgroup must contain both assignment arms.')
    site_indicators = pd.get_dummies(
        sample['site'], prefix='site', drop_first=True, dtype=float
    )
    design = pd.concat([sample[['treatment']], site_indicators], axis=1)
    design = sm.add_constant(design, has_constant='add').astype(float)
    matrix = design.to_numpy()
    if len(sample) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError('The subgroup design must have full rank and residual degrees of freedom.')
    leverage = np.sum(matrix * (matrix @ np.linalg.inv(matrix.T @ matrix)), axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('HC2 inference is undefined with unit-leverage observations.')

    model = sm.OLS(sample['food_insecure_m12'], design).fit(
        cov_type='HC2', use_t=True
    )
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()) or result['std_error'] <= 0:
        raise ValueError('The chosen test did not yield finite, nondegenerate inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
