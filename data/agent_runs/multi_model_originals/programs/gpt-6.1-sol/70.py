#!/usr/bin/env python3
"""Fixed analysis: ITT effect on hours_m12 among participants at site 1.

OLS with an intercept and treatment, HC2 standard errors, and a two-sided
Student t test using residual degrees of freedom. Site indicators are constant
within this subgroup. No baseline adjustment and no search are performed.

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path, usecols=['site', 'treatment', 'hours_m12'])
    sample = data.loc[data['site'].eq(1), ['treatment', 'hours_m12']].copy()
    if sample.empty:
        raise ValueError('The dataset has no participants at site 1.')
    sample = sample.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(sample.to_numpy(dtype=float)).all():
        raise ValueError('Treatment and hours_m12 must be observed and finite at site 1.')
    if not sample['treatment'].isin([0, 1]).all():
        raise ValueError('Treatment must be coded 0 or 1.')
    counts = sample['treatment'].value_counts()
    if any(counts.get(arm, 0) < 2 for arm in (0, 1)):
        raise ValueError('HC2 inference requires at least two participants per arm at site 1.')
    design = sm.add_constant(sample[['treatment']].astype(float), has_constant='add')
    fit = sm.OLS(sample['hours_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()) or result['std_error'] <= 0:
        raise ValueError('The supplied sample does not support finite, nondegenerate inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
