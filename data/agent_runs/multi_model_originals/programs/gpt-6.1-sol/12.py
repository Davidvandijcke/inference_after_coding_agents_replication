#!/usr/bin/env python3
"""Fixed Pathways to Work analysis: six-month distress among women.

Usage: python score.py --data data.csv --output result.json
Requires pandas, scipy, and statsmodels (and their dependencies).

The selected intention-to-treat analysis is OLS of distress_m6 on treatment,
an intercept, and categorical site indicators, restricted to female == 1.
Inference uses HC2 covariance and a two-sided Student t test with residual
OLS degrees of freedom. No outcome, subgroup, or model search is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    """Return the single frozen treatment-effect test on the supplied data."""
    required = ['female', 'treatment', 'site', 'distress_m6']
    data = pd.read_csv(data_path, usecols=required)
    if data[required].isna().any().any():
        raise ValueError('Required analysis columns must have no missing values.')
    if not data['female'].isin([0, 1]).all():
        raise ValueError('female must contain only 0 and 1.')
    women = data.loc[data['female'].eq(1)].copy()
    if set(women['treatment'].unique()) != {0, 1}:
        raise ValueError('The female subgroup must contain both treatment arms.')
    numeric = women[['treatment', 'distress_m6']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('Treatment and distress must be finite numeric values.')

    model = smf.ols('distress_m6 ~ treatment + C(site)', data=women)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Treatment and site effects are not separately identifiable.')
    if design.shape[0] <= design.shape[1]:
        raise ValueError('There are no residual degrees of freedom.')
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('Unit leverage prevents HC2 inference.')

    fitted = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fitted.params['treatment']),
        'std_error': float(fitted.bse['treatment']),
        'p_value': float(fitted.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The fixed analysis did not produce finite results.')
    if result['std_error'] <= 0:
        raise ValueError('A positive standard error is required for inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
