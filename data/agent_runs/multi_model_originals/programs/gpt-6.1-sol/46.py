#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis; no exploratory search.

Estimand: treatment assignment's effect on earnings_m6 among participants with
employed_baseline == 1. Regress earnings_m6 on an intercept, treatment, and
categorical site indicators. Report the treatment coefficient, HC2 standard
error, and two-sided Student t p-value with OLS residual degrees of freedom.

Requires numpy, pandas, scipy, and statsmodels.
Usage: python score.py --data data.csv --output chosen_result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data):
    required = ['site', 'treatment', 'employed_baseline', 'earnings_m6']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data[required].isna().any().any():
        raise ValueError('Required columns must be fully observed.')
    for column in ['treatment', 'employed_baseline']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be coded 0 or 1.')
    subgroup = data.loc[data['employed_baseline'].eq(1)].copy()
    if subgroup.empty or subgroup['treatment'].nunique() != 2:
        raise ValueError('The baseline-employed subgroup needs both treatment arms.')
    if not np.isfinite(subgroup['earnings_m6'].to_numpy(dtype=float)).all():
        raise ValueError('earnings_m6 must contain finite numeric values.')
    model = smf.ols('earnings_m6 ~ treatment + C(site)', data=subgroup,
                    missing='raise')
    if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
        raise ValueError('Treatment and site design matrix is not full rank.')
    if model.exog.shape[0] <= model.exog.shape[1]:
        raise ValueError('The model needs positive residual degrees of freedom.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen model did not produce finite inference.')
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
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')


if __name__ == '__main__':
    main()
