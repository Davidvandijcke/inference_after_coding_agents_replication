#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on 12-month employment.

Dependencies: pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.data).resolve() == Path(args.output).resolve():
        parser.error('Output must differ from the input data path.')

    data = pd.read_csv(args.data, usecols=['site', 'treatment', 'employed_m12'])
    if data.empty or data.isna().any().any():
        raise ValueError('The selected analysis requires complete, nonempty data.')
    if not data['treatment'].isin([0, 1]).all() or data['treatment'].nunique() != 2:
        raise ValueError('Treatment must contain both 0 and 1, and no other values.')
    if not data['employed_m12'].isin([0, 1]).all():
        raise ValueError('employed_m12 must be binary (0 or 1).')
    if not data.groupby('site')['treatment'].nunique().eq(2).all():
        raise ValueError('Each site must contain both treatment groups.')

    sites = pd.get_dummies(data['site'], prefix='site', drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([data[['treatment']], sites], axis=1),
                             has_constant='add').astype(float)
    if len(data) <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom.')
    # HC2 covariance, with a two-sided Student t test using residual df.
    model = sm.OLS(data['employed_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    with open(args.output, 'w', encoding='utf-8') as output:
        json.dump(result, output, allow_nan=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
