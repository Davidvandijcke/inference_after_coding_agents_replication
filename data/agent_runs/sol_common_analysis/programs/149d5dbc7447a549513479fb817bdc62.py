#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels (and their dependencies).
Model: employed_m12 ~ treatment + C(site) + employed_baseline.
Inference: HC2 covariance; two-sided Student t test with OLS residual df.
Estimates and standard errors are in probability units, not percentage points.
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

    data = pd.read_csv(args.data)
    result = smf.ols(
        'employed_m12 ~ treatment + C(site) + employed_baseline',
        data=data,
        missing='raise',
    ).fit(cov_type='HC2', use_t=True)
    output = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    args.output.write_text(json.dumps(output, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
