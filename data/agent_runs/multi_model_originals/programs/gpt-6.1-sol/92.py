#!/usr/bin/env python3
"""Run the fixed, exploratory Pathways to Work analysis.

Chosen estimand: assignment effect on food_insecure_m6 among female == 0.
Model: intercept + treatment + categorical site indicators, with no additional
covariates. Inference: HC2 robust SE and a two-sided Student t-test using the
OLS residual degrees of freedom. This program never selects a test from data.

Dependencies: numpy, pandas, statsmodels (and their dependencies).
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    frame = pd.read_csv(data_path)
    required = ['female', 'site', 'treatment', 'food_insecure_m6']
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if frame[required].isna().any().any():
        raise ValueError('Required analysis columns must be fully observed.')
    for col in ['female', 'treatment', 'food_insecure_m6']:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f'{col} must contain only 0 and 1.')

    subset = frame.loc[frame.female.eq(0)].copy()
    if subset.empty or subset.treatment.nunique() != 2:
        raise ValueError('female=0 subgroup must contain both treatment arms.')
    design = pd.concat([
        pd.Series(1., index=subset.index, name='intercept'),
        subset[['treatment']].astype(float),
        pd.get_dummies(subset.site, prefix='site', drop_first=True, dtype=float),
    ], axis=1)
    if (len(subset) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]):
        raise ValueError('Chosen regression is not estimable with positive residual df.')
    fit = sm.OLS(subset.food_insecure_m6.astype(float), design).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('Chosen HC2 t-test produced non-finite results.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
