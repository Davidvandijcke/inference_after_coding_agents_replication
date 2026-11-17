#!/usr/bin/env python3
"""Run the fixed, selected Pathways to Work analysis.

Estimand: effect of program assignment on debt_collections_m12 among
participants with employed_baseline == 0. OLS includes an intercept and
categorical site indicators, with HC2 standard errors and a two-sided
Student t test using residual degrees of freedom. Effects are proportions
(multiply estimate and std_error by 100 for percentage points).

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, scipy, and statsmodels. No search is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    data = pd.read_csv(data_path)
    required = ['employed_baseline', 'treatment', 'site', 'debt_collections_m12']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data[required].isna().any().any():
        raise ValueError('Required analysis columns must be fully observed.')
    for column in ['employed_baseline', 'treatment', 'debt_collections_m12']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1.')

    subgroup = data.loc[data['employed_baseline'] == 0].copy()
    if set(subgroup['treatment'].unique()) != {0, 1}:
        raise ValueError('The selected subgroup must contain both treatment arms.')
    model = smf.ols(
        'debt_collections_m12 ~ treatment + C(site)', data=subgroup,
        missing='raise',
    )
    design = model.exog
    if (np.linalg.matrix_rank(design) != design.shape[1]
            or design.shape[0] <= design.shape[1]):
        raise ValueError('The selected regression must have full rank and residual degrees of freedom.')
    result = model.fit(cov_type='HC2', use_t=True)
    output = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError('The selected analysis produced nonfinite inference.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    output = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
