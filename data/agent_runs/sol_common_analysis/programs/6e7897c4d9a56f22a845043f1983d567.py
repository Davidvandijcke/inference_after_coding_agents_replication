#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Uses OLS with site fixed effects, HC2
standard errors, and a two-sided t-test with residual degrees of freedom.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')

    data = pd.read_csv(args.data, usecols=['site', 'treatment', 'employed_m12'])
    if data.empty or data.isna().any().any():
        parser.error('Analysis columns must contain complete observations')
    for column in ['treatment', 'employed_m12']:
        if not data[column].isin([0, 1]).all():
            parser.error(f'{column} must contain only 0 and 1')
    if data.groupby('site')['treatment'].nunique().ne(2).any():
        parser.error('Each site must contain both treatment groups')

    fit = smf.ols('employed_m12 ~ treatment + C(site)', data=data).fit(
        cov_type='HC2', use_t=True
    )
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    # Strict JSON also rejects undefined estimates or standard errors.
    payload = json.dumps(result, allow_nan=False, indent=2) + '\n'
    args.output.write_text(payload, encoding='utf-8')


if __name__ == '__main__':
    main()
