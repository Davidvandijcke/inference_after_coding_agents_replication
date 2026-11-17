#!/usr/bin/env python3
"""Frozen Pathways to Work analysis; no outcome or subgroup search.

Estimate the intention-to-treat effect on training_m12 among female == 0:
OLS(training_m12 ~ treatment + C(site)), HC2 covariance, two-sided t test
with residual degrees of freedom. Estimates are in probability units.

Requires numpy, pandas, and statsmodels (and their dependencies).
Usage: python score.py --data data.csv --output chosen_result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    columns = ['female', 'site', 'treatment', 'training_m12']
    data = pd.read_csv(data_path, usecols=columns)
    if data[columns].isna().any().any():
        raise ValueError('Required analysis columns must have no missing values.')
    for column in ['female', 'treatment', 'training_m12']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1.')
    sample = data.loc[data['female'].eq(0)].copy()
    if sample.empty or sample['treatment'].nunique() != 2:
        raise ValueError('The female=0 subgroup must contain both treatment arms.')

    model = smf.ols('training_m12 ~ treatment + C(site)', data=sample)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Treatment and site effects are not separately identified.')
    if len(sample) <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom.')
    leverage = np.sum(design * (design @ np.linalg.inv(design.T @ design)), axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('Unit leverage makes HC2 inference undefined.')

    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The selected regression produced nonfinite results.')
    if result['std_error'] <= 0:
        raise ValueError('The selected test has no positive estimated standard error.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
