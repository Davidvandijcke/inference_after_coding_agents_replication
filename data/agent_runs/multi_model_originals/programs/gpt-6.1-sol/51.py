#!/usr/bin/env python3
"""Estimate the fixed Pathways to Work analysis; does not select an analysis.

Chosen test: intention-to-treat effect on debt_collections_m12 among all
participants, using OLS with an intercept, treatment, and site indicators.
Inference: HC2 standard error and two-sided Student t test, df = n - rank(X).
Estimate and standard error are in probability units (multiply by 100 for pp).
Dependencies: numpy, pandas, scipy, statsmodels.

Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    columns = ['site', 'treatment', 'debt_collections_m12']
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError('The chosen analysis requires complete, nonempty data.')
    for name in ['treatment', 'debt_collections_m12']:
        if not data[name].isin([0, 1]).all():
            raise ValueError(f'{name} must contain only 0 and 1.')
    if data.treatment.nunique() != 2:
        raise ValueError('Both treatment arms are required.')

    design = pd.concat([
        data[['treatment']].astype(float),
        pd.get_dummies(data.site, prefix='site', drop_first=True, dtype=float),
    ], axis=1)
    design = sm.add_constant(design, has_constant='add')
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank != design.shape[1] or len(data) <= rank:
        raise ValueError('The regression needs full rank and positive residual degrees of freedom.')
    fit = sm.OLS(data.debt_collections_m12.astype(float), design).fit(
        cov_type='HC2', use_t=True,
    )
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()) or result['std_error'] <= 0:
        raise ValueError('The regression did not yield a finite, positive standard error.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='CSV dataset with the supplied column names')
    parser.add_argument('--output', required=True, help='Destination JSON file')
    args = parser.parse_args()
    source = Path(args.data).resolve()
    destination = Path(args.output).resolve()
    if destination == source:
        raise ValueError('Output must not overwrite the input dataset.')
    result = analyze(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
