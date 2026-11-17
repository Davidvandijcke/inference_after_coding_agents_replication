#!/usr/bin/env python3
"""Run the fixed, exploratory Site 1 six-month earnings comparison.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, statsmodels (and their dependencies).

The test is the intention-to-treat coefficient for earnings_m6 among site == 1,
with an intercept and treatment indicator, HC2 standard errors, and a two-sided
Student t test using OLS residual degrees of freedom. Site indicators are
constant within this subgroup and hence redundant. There are no additional
baseline covariates and no search, transformation, or multiplicity adjustment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    frame = pd.read_csv(data_path, usecols=['site', 'treatment', 'earnings_m6'])
    frame['site'] = pd.to_numeric(frame['site'], errors='raise')
    frame = frame.loc[frame['site'] == 1].copy()
    for name in ['treatment', 'earnings_m6']:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
        if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(f'{name} must be observed and finite within Site 1')
    if not frame['treatment'].isin([0, 1]).all():
        raise ValueError('treatment must be 0 or 1 within Site 1')
    counts = frame['treatment'].value_counts()
    if any(counts.get(arm, 0) < 2 for arm in [0, 1]):
        raise ValueError('Site 1 must contain at least two participants per arm')

    design = sm.add_constant(frame[['treatment']].astype(float), has_constant='add')
    fit = sm.OLS(frame['earnings_m6'].astype(float), design).fit(
        cov_type='HC2', use_t=True
    )
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen test did not produce finite results')
    if result['std_error'] <= 0:
        raise ValueError('The chosen test requires a positive standard error')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
