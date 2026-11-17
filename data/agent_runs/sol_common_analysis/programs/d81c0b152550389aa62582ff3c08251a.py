#!/usr/bin/env python3
"""Estimate the 12-month earnings intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels (and their dependencies).
The only analysis is OLS earnings_m12 ~ treatment + site fixed effects,
with HC2 standard errors and a two-sided residual-df t-test.
The p-value is nominal; it does not adjust for exploratory outcome selection.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ['earnings_m12', 'treatment', 'site']
    if data[required].isna().any().any():
        raise ValueError('Analysis columns must be fully observed.')
    if set(data['treatment'].unique()) != {0, 1}:
        raise ValueError('Treatment must contain both 0 and 1.')

    fit = smf.ols(
        'earnings_m12 ~ treatment + C(site)', data=data, missing='raise'
    ).fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
