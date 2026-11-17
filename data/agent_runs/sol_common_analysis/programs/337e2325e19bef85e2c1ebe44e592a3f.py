#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, statsmodels.

The sole analysis is OLS: employed_m12 ~ treatment + C(site), with
HC2 covariance and a two-sided Student t-test using residual degrees of freedom.
The estimate and standard error are in probability units (1.0 = 100 pp).
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

    data = pd.read_csv(args.data, usecols=['employed_m12', 'treatment', 'site'])
    if data.empty or data.isna().any().any():
        parser.error('The analysis requires nonempty, complete observations.')
    if not data['employed_m12'].isin([0, 1]).all():
        parser.error('employed_m12 must be binary (0/1).')
    if set(data['treatment'].unique()) != {0, 1}:
        parser.error('treatment must contain both 0 and 1, and no other values.')

    model = smf.ols('employed_m12 ~ treatment + C(site)', data=data).fit(
        cov_type='HC2', use_t=True
    )
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding='utf-8')


if __name__ == '__main__':
    main()
