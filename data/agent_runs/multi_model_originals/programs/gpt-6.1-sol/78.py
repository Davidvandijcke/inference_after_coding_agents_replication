#!/usr/bin/env python3
"""Run the fixed, exploratory Pathways to Work finding.

Analysis: participants younger than 35 at random assignment; six-month
self-rated health regressed on treatment, an intercept, and site indicators.
Inference: HC2 standard error and two-sided Student t test with OLS residual
degrees of freedom. The p-value is nominal, without selection adjustment.

Requires numpy, pandas, scipy, and statsmodels. This program performs no search.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data):
    required = ['age', 'site', 'treatment', 'health_m6']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if data[required].isna().any().any():
        raise ValueError('The chosen analysis requires complete data.')
    for column in required:
        if not np.isfinite(pd.to_numeric(data[column], errors='raise')).all():
            raise ValueError(f'{column} must contain finite numeric values.')
    if not data.treatment.isin([0, 1]).all():
        raise ValueError('treatment must be coded 0 or 1.')
    subgroup = data.loc[data.age < 35].copy()
    if subgroup.treatment.nunique() != 2:
        raise ValueError('The under-35 subgroup must contain both assignment arms.')
    design = sm.add_constant(pd.concat([
        subgroup[['treatment']].astype(float),
        pd.get_dummies(subgroup.site, prefix='site', drop_first=True, dtype=float),
    ], axis=1), has_constant='add')
    if len(design) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('The subgroup design must have full rank and residual degrees of freedom.')
    # An HC2 variance is undefined for a perfectly fitted individual observation.
    matrix = design.to_numpy()
    leverage = np.sum(matrix * np.linalg.pinv(matrix).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('HC2 requires every observation to have leverage below one.')
    model = sm.OLS(subgroup.health_m6.astype(float), design).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen analysis did not yield finite inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
