#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis, without exploratory selection.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels.

Estimand: assignment effect on education/job-training enrollment at month 6,
within participants employed at baseline. Units are probability units (0.01 is
one percentage point). Model: OLS on an intercept, treatment, and site indicators.
Inference: HC2 covariance, two-sided Student t test, df = n - rank(X).
The returned p_value is the nominal p-value for this fixed coefficient, with no
search or multiplicity correction. See analysis/report.md for selection context.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    required = ['site', 'treatment', 'employed_baseline', 'training_m6']
    data = pd.read_csv(data_path, usecols=required)
    if data[required].isna().any().any():
        raise ValueError('Required analysis columns must have no missing values.')
    for column in ['treatment', 'employed_baseline', 'training_m6']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1.')

    # Both the subgroup and the endpoint are fixed, regardless of input results.
    subgroup = data.loc[data['employed_baseline'].eq(1)].copy()
    if subgroup.empty or subgroup['treatment'].nunique() != 2:
        raise ValueError('The baseline-employed subgroup needs both treatment arms.')
    site_indicators = pd.get_dummies(
        subgroup['site'], prefix='site', drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([subgroup[['treatment']].astype(float), site_indicators], axis=1),
        has_constant='add',
    )
    matrix = design.to_numpy()
    if len(subgroup) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError('The selected regression requires a full-rank design and residual degrees of freedom.')
    leverage = np.sum(matrix * np.linalg.pinv(matrix).T, axis=1)
    if np.any(1.0 - leverage <= 1e-12):
        raise ValueError('HC2 is undefined for observations with leverage one.')

    fitted = sm.OLS(subgroup['training_m6'].astype(float), design).fit(
        cov_type='HC2', use_t=True
    )
    result = {
        'estimate': float(fitted.params['treatment']),
        'std_error': float(fitted.bse['treatment']),
        'p_value': float(fitted.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()) or result['std_error'] <= 0:
        raise ValueError('The selected regression did not yield finite, nondegenerate inference.')
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
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
