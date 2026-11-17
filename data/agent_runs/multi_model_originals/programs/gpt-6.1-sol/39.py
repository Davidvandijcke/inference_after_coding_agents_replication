#!/usr/bin/env python3
"""Run the fixed, selected Pathways to Work analysis.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, statsmodels (and their dependencies).

Estimand: full-sample, baseline-adjusted intention-to-treat effect on employment
at six months, in probability units (0.02 means 2 percentage points).
Inference: OLS, HC2 covariance, two-sided Student t test with OLS residual df.
The outcome, population, covariates, and inference are fixed; no search is run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    'employed_m6 ~ treatment + C(site) + age + female + parent + C(education)'
    ' + household_size + employed_baseline + earnings_baseline'
    ' + months_unemployed_baseline'
)
REQUIRED = [
    'employed_m6', 'treatment', 'site', 'age', 'female', 'parent', 'education',
    'household_size', 'employed_baseline', 'earnings_baseline',
    'months_unemployed_baseline',
]


def analyze(data_path):
    data = pd.read_csv(data_path)
    missing = sorted(set(REQUIRED) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    data = data.loc[:, REQUIRED].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Required variables must be finite and fully observed.')
    for column in ['employed_m6', 'treatment', 'female', 'parent', 'employed_baseline']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1.')
    for column, levels in [('site', [1, 2, 3, 4]), ('education', [1, 2, 3, 4])]:
        if not data[column].isin(levels).all():
            raise ValueError(f'{column} must contain only {levels}.')
        data[column] = pd.Categorical(data[column], categories=levels)
    if data['treatment'].nunique() != 2:
        raise ValueError('Both treatment arms are required.')

    model = smf.ols(FORMULA, data=data, missing='raise')
    design = model.exog
    treatment_index = model.exog_names.index('treatment')
    nuisance = np.delete(design, treatment_index, axis=1)
    if np.linalg.matrix_rank(design) <= np.linalg.matrix_rank(nuisance):
        raise ValueError('Treatment effect is not identified given the covariates.')
    if len(data) <= np.linalg.matrix_rank(design):
        raise ValueError('Positive residual degrees of freedom are required.')
    if np.any(np.sum(design * np.linalg.pinv(design).T, axis=1) >= 1 - 1e-12):
        raise ValueError('HC2 inference requires leverage strictly below one.')

    result = model.fit(cov_type='HC2', use_t=True)
    output = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError('The fixed analysis did not produce finite inference.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    output = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
