#!/usr/bin/env python3
"""Frozen analysis: ITT effect on training enrollment at six months.

Usage: python score.py --data data.csv --output result.json

Fits OLS training_m6 ~ treatment + C(site) to all participants, with an
intercept, HC2 covariance, and a two-sided Student t test using residual
degrees of freedom. Estimates and standard errors are in probability units
(multiply by 100 for percentage points). No subgroup or outcome search occurs.

Dependencies: pandas, numpy, statsmodels (and its scipy dependency).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ['training_m6', 'treatment', 'site']
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data.empty or data[required].isna().any().any():
        raise ValueError('The analysis requires nonempty, complete observations.')
    if not data['training_m6'].isin([0, 1]).all():
        raise ValueError('training_m6 must be coded 0 or 1.')
    if set(data['treatment'].unique()) != {0, 1}:
        raise ValueError('treatment must contain both 0 and 1, and no other values.')

    # Converting numeric fields ensures the named coefficient is treatment.
    data = data[required].copy()
    data['treatment'] = data['treatment'].astype(float)
    data['training_m6'] = data['training_m6'].astype(float)
    model = smf.ols('training_m6 ~ treatment + C(site)', data=data)
    design = model.exog
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('The treatment effect is not identified by this dataset.')
    if design.shape[0] <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The requested HC2 t test is undefined for this dataset.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    try:
        result = analyze(args.data)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
