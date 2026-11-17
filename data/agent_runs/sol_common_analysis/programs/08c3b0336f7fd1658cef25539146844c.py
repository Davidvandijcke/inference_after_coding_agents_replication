#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment, adjusting for site.

Requires numpy, pandas, and statsmodels. Uses OLS, HC2 standard errors,
and a two-sided Student t-test with residual degrees of freedom.
Estimates are in probability units (0.01 = one percentage point).
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
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')

    data = pd.read_csv(args.data)
    required = ['site', 'treatment', 'employed_m12']
    if any(column not in data for column in required):
        parser.error('data must contain site, treatment, and employed_m12')
    if data[required].isna().any().any():
        parser.error('analysis columns must have no missing values')
    if set(data['treatment'].unique()) != {0, 1}:
        parser.error('treatment must contain both 0 and 1')
    if not data['employed_m12'].isin([0, 1]).all():
        parser.error('employed_m12 must contain only 0 and 1')

    sites = pd.get_dummies(data['site'], prefix='site', drop_first=True,
                           dtype=float)
    design = sm.add_constant(pd.concat([data[['treatment']], sites], axis=1),
                             has_constant='add').astype(float)
    if (len(data) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]):
        parser.error('analysis requires a full-rank design and positive residual degrees of freedom')
    fit = sm.OLS(data['employed_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n',
                           encoding='utf-8')


if __name__ == '__main__':
    main()
