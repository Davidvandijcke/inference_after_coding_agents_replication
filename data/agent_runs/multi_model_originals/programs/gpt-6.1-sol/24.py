#!/usr/bin/env python3
"""Frozen Pathways to Work analysis: 12-month training enrollment among women.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, scipy, and statsmodels.

OLS: training_m12 ~ intercept + treatment + categorical site indicators,
restricted to female == 1. HC2 standard errors and a two-sided Student t test
with residual degrees of freedom. Output is in probability units. The p-value
is nominal, without correction for the exploratory search that chose this test.
No outcome, subgroup, follow-up, or model selection occurs here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ['female', 'site', 'treatment', 'training_m12']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        raise ValueError('Required analysis columns must have no missing values.')
    for column in ['female', 'treatment', 'training_m12']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1.')

    women = data.loc[data['female'] == 1]
    if women.empty or women['treatment'].nunique() != 2:
        raise ValueError('The female subgroup must include both treatment arms.')
    sites = pd.get_dummies(women['site'], prefix='site', drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([women[['treatment']].astype(float), sites], axis=1),
        has_constant='add',
    )
    matrix = design.to_numpy()
    if len(women) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError('The selected regression must have full rank and positive residual degrees of freedom.')
    model = sm.OLS(women['training_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True,
    )
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()) or result['std_error'] <= 0:
        raise ValueError('The selected regression cannot produce a finite, positive standard error and valid inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Input CSV with the codebook columns.')
    parser.add_argument('--output', type=Path, required=True, help='Destination JSON file.')
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
