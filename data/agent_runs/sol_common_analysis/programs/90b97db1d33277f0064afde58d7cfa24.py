#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, and statsmodels (and its scipy dependency).
The only fitted model is employed_m12 ~ treatment + C(site), with HC2
standard errors and a two-sided Student t test using residual degrees of freedom.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('Output must differ from the input data path.')

    data = pd.read_csv(args.data, usecols=['employed_m12', 'treatment', 'site'])
    if data.empty or data.isna().any().any():
        parser.error('The analysis columns must contain complete observations.')
    for column in ['employed_m12', 'treatment']:
        if not data[column].isin([0, 1]).all():
            parser.error(f'{column} must be coded 0 or 1.')
    if data['treatment'].nunique() != 2:
        parser.error('Both treatment arms are required.')

    model = smf.ols('employed_m12 ~ treatment + C(site)', data=data,
                    missing='raise')
    if (np.linalg.matrix_rank(model.exog) != model.exog.shape[1]
            or len(data) <= model.exog.shape[1]):
        parser.error('The regression requires a full-rank design and positive residual degrees of freedom.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        parser.error('The specified regression did not yield finite inference.')
    args.output.write_text(json.dumps(result, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
