#!/usr/bin/env python3
"""Run the frozen Pathways to Work analysis; no outcome/subgroup search.

Target: intention-to-treat effect on training_m12 among age >= 50.
Model: OLS with an intercept, treatment, and site indicators.
Inference: HC2 standard error and a two-sided Student t test using the
OLS residual degrees of freedom. Estimates are in probability units.

Dependencies: numpy, pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    """Return only the frozen coefficient, HC2 SE, and t-test p-value."""
    data = pd.read_csv(data_path)
    required = ['age', 'site', 'treatment', 'training_m12']
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data[required].isna().any().any():
        raise ValueError('The chosen analysis requires complete relevant columns.')
    for column in ['age', 'treatment', 'training_m12']:
        data[column] = pd.to_numeric(data[column], errors='raise')
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f'{column} must contain finite numeric values.')
    for column in ['treatment', 'training_m12']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be coded 0 or 1.')

    subgroup = data.loc[data['age'].ge(50)].copy()
    if subgroup.empty or subgroup['treatment'].nunique() != 2:
        raise ValueError('The age >= 50 subgroup must contain both treatment arms.')

    design = pd.concat([
        subgroup[['treatment']].astype(float),
        pd.get_dummies(subgroup['site'], prefix='site',
                       drop_first=True, dtype=float),
    ], axis=1)
    design = sm.add_constant(design, has_constant='add')
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError('The treatment effect is not identifiable with site controls.')
    if len(subgroup) <= design.shape[1]:
        raise ValueError('The subgroup has no residual degrees of freedom.')

    model = sm.OLS(subgroup['training_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen test has undefined inference on this dataset.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    try:
        result = chosen_analysis(args.data)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')


if __name__ == '__main__':
    main()
