#!/usr/bin/env python3
"""Fixed analysis: assignment effect on 12-month employment within site 3.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Estimates OLS with an intercept and treatment,
HC2 standard errors, and a two-sided Student t test with n - 2 residual degrees
of freedom. Within one site, site fixed effects are absorbed by the intercept.
The estimate is in probability units; multiply by 100 for percentage points.
The p-value is nominal: this program performs no exploratory search or
multiple-testing correction.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ['site', 'treatment', 'employed_m12']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    site = pd.to_numeric(data['site'], errors='raise')
    if site.isna().any():
        raise ValueError('site must be observed for every row')
    sample = data.loc[site.eq(3), ['treatment', 'employed_m12']].apply(
        pd.to_numeric, errors='raise')
    if sample.empty or sample.isna().any().any():
        raise ValueError('Site 3 must have complete treatment and outcome data')
    for column in sample:
        if not sample[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1')
    if any(int(sample.treatment.eq(arm).sum()) < 2 for arm in (0, 1)):
        raise ValueError('Site 3 needs at least two participants in each arm for HC2')
    design = sm.add_constant(sample[['treatment']].astype(float), has_constant='add')
    fitted = sm.OLS(sample.employed_m12.astype(float), design).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fitted.params['treatment']),
        'std_error': float(fitted.bse['treatment']),
        'p_value': float(fitted.pvalues['treatment']),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('The supplied data do not yield finite OLS/HC2 inference')
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
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
