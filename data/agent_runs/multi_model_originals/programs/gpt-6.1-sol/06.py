#!/usr/bin/env python3
"""Fixed Pathways to Work analysis: 12-month employment, all participants.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, statsmodels (and its scipy dependency).

OLS: employed_m12 ~ treatment + C(site), with an intercept.
Inference: HC2 robust standard error; two-sided Student t test with
residual degrees of freedom. No baseline covariates or subgroup selection.
The estimate and standard error are in probability units (0.01 = 1 pp).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    frame = pd.read_csv(data_path)
    required = ['employed_m12', 'treatment', 'site']
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    frame = frame[required].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError('The analysis requires nonempty, complete observations.')
    for name in ['employed_m12', 'treatment']:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f'{name} must contain only 0 and 1.')
    if frame.treatment.nunique() != 2:
        raise ValueError('Both treatment arms are required.')

    model = smf.ols('employed_m12 ~ treatment + C(site)', data=frame,
                    missing='raise')
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('The site-adjusted treatment effect is not identified.')
    if len(frame) <= design.shape[1]:
        raise ValueError('Positive residual degrees of freedom are required.')
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('HC2 inference is undefined for unit-leverage observations.')
    result = model.fit(cov_type='HC2', use_t=True)
    values = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError('The fitted model did not yield finite inference.')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    values = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')


if __name__ == '__main__':
    main()
