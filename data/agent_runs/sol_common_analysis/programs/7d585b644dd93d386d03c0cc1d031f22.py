#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only employed_m12 ~ treatment + C(site),
with an intercept, HC2 standard errors, and a two-sided residual-df t-test.
The estimate and standard error are in probability units (not percentage points).
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

    data = pd.read_csv(args.data, usecols=['employed_m12', 'treatment', 'site'])
    if data.empty or data.isna().any().any():
        raise ValueError('Analysis columns must be nonempty and fully observed.')
    if set(data['treatment'].unique()) != {0, 1}:
        raise ValueError('Treatment must contain both 0 and 1 only.')
    if not data['employed_m12'].isin([0, 1]).all():
        raise ValueError('employed_m12 must contain only 0 and 1.')

    fit = smf.ols('employed_m12 ~ treatment + C(site)', data=data,
                  missing='raise').fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
