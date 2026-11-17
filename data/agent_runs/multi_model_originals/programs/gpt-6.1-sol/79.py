#!/usr/bin/env python3
"""Frozen analysis: baseline-adjusted intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels.

OLS uses an intercept, treatment, site indicators, age, female, parent,
household_size, baseline employment, baseline earnings, baseline months
unemployed, and categorical education indicators. HC2 standard errors and a
two-sided Student t test with residual degrees of freedom follow CODEBOOK.md.
The estimate and standard error are in probability units (0.01 = 1 percentage
point). This program performs no outcome, subgroup, or specification search.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASELINE = [
    'age', 'female', 'parent', 'household_size', 'employed_baseline',
    'earnings_baseline', 'months_unemployed_baseline',
]
OUTCOME = 'employed_m12'


def analyze(data):
    required = ['treatment', 'site', 'education', OUTCOME] + BASELINE
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    d = data[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(d.to_numpy(dtype=float)).all():
        raise ValueError('Analysis variables must be observed and finite in every row.')
    if set(d.treatment.unique()) != {0, 1}:
        raise ValueError('Treatment must contain both 0 and 1 and no other values.')
    if not d[OUTCOME].isin([0, 1]).all():
        raise ValueError('employed_m12 must be binary (0 or 1).')
    x = pd.concat([
        pd.Series(1.0, index=d.index, name='const'),
        d[['treatment'] + BASELINE].astype(float),
        pd.get_dummies(d.site, prefix='site', drop_first=True, dtype=float),
        pd.get_dummies(d.education, prefix='education', drop_first=True, dtype=float),
    ], axis=1)
    # Verify treatment is estimable even if another dataset has redundant controls.
    controls = x.drop(columns='treatment').to_numpy()
    if np.linalg.matrix_rank(x.to_numpy()) == np.linalg.matrix_rank(controls):
        raise ValueError('Treatment is collinear with the specified controls.')
    fit = sm.OLS(d[OUTCOME].astype(float), x).fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if fit.df_resid <= 0 or not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Insufficient information for the specified HC2 t test.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = analyze(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
