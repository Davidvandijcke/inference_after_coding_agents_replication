#!/usr/bin/env python3
"""Frozen Pathways to Work hypothesis; no exploration or model selection.

Estimate the intention-to-treat effect on food insecurity at 12 months
among participants not employed at baseline. Fit OLS with an intercept,
treatment, and categorical site indicators; use HC2 standard errors and
a two-sided Student t test with OLS residual degrees of freedom.

The estimate and standard error are in proportions (0.01 = 1 percentage
point). A positive estimate means greater food insecurity under assignment
to the program. Dependencies: numpy, pandas, scipy, statsmodels.

Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    data = pd.read_csv(data_path)
    required = ['site', 'treatment', 'employed_baseline', 'food_insecure_m12']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data[required].isna().any().any():
        raise ValueError('Required columns must be fully observed.')
    for column in ['treatment', 'employed_baseline', 'food_insecure_m12']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be coded 0 or 1.')
    subgroup = data.loc[data.employed_baseline.eq(0)]
    if subgroup.treatment.nunique() != 2:
        raise ValueError('The baseline-not-employed subgroup needs both treatment arms.')
    model = smf.ols('food_insecure_m12 ~ treatment + C(site)', data=subgroup)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Treatment and site effects are not separately identifiable.')
    if model.df_resid <= 0:
        raise ValueError('Insufficient residual degrees of freedom.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The selected analysis produced nonfinite inference.')
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
