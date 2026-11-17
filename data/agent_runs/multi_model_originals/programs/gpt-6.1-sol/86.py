#!/usr/bin/env python3
"""Frozen analysis: the full-sample treatment effect on employed_m12.

Usage: python score.py --data data.csv --output result.json
Requires Python 3, numpy, pandas, scipy, and statsmodels.

OLS with an intercept, treatment, and categorical site indicators; HC2 standard
error and a two-sided t-test with residual degrees of freedom. No optional
baseline covariates, subgroup selection, outcome search, or multiplicity
adjustment is performed. Estimate and standard error are in probability units.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ['treatment', 'site', 'employed_m12']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data[required].isna().any().any():
        raise ValueError('The chosen analysis requires complete treatment, site, and employed_m12 data.')
    if not data.treatment.isin([0, 1]).all() or data.treatment.nunique() != 2:
        raise ValueError('treatment must contain both 0 and 1, with no other values.')
    if not data.employed_m12.isin([0, 1]).all():
        raise ValueError('employed_m12 must be coded 0 or 1.')

    sites = pd.get_dummies(data.site, prefix='site', drop_first=True, dtype=float)
    design = pd.concat([data[['treatment']], sites], axis=1)
    design = sm.add_constant(design, has_constant='add').astype(float)
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('Insufficient observations or treatment variation to estimate the model.')

    result = sm.OLS(data.employed_m12.astype(float), design).fit(cov_type='HC2', use_t=True)
    output = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in output.values()) or output['std_error'] <= 0:
        raise ValueError('The HC2 standard error and t-test are not defined for this dataset.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
