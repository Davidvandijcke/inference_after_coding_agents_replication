#!/usr/bin/env python3
"""Frozen exploratory test: six-month earnings effect among site 3 participants.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels (and their numpy/scipy dependencies).
No outcome, subgroup, or model selection is performed.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data_path):
    data = pd.read_csv(data_path)
    required = ['site', 'treatment', 'earnings_m6']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    for column in required:
        data[column] = pd.to_numeric(data[column], errors='raise')
    if data['site'].isna().any():
        raise ValueError('Site must be observed for every participant.')
    sample = data.loc[data['site'].eq(3), required].copy()
    if sample.empty:
        raise ValueError('The chosen analysis requires participants at site 3.')
    if not all(math.isfinite(float(value)) for value in sample.to_numpy().ravel()):
        raise ValueError('The selected analysis columns must be finite and complete.')
    if not sample.treatment.isin([0, 1]).all():
        raise ValueError('Treatment must be coded 0 or 1.')
    counts = sample.treatment.value_counts()
    if any(counts.get(arm, 0) < 2 for arm in (0, 1)):
        raise ValueError('HC2 inference requires at least two participants in each arm.')
    # Within site 3, C(site) is constant: this is an intercept and treatment.
    # use_t=True gives two-sided Student t inference with OLS residual df (n - 2).
    fit = smf.ols('earnings_m6 ~ treatment + C(site)', data=sample).fit(
        cov_type='HC2', use_t=True
    )
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('The selected data do not permit finite HC2 t inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
