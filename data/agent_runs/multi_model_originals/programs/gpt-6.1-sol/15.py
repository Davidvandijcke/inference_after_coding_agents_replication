"""Rerun the fixed, exploratory Pathways to Work finding.

Chosen estimand: intention-to-treat effect on employed_m12 within female == 1,
estimated by OLS with an intercept and site fixed effects. The standard error
is HC2; p_value is the nominal two-sided t-test with residual degrees of freedom.
The script performs no outcome, subgroup, cutoff, or specification search.

Requires numpy, pandas, scipy, and statsmodels.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ['female', 'treatment', 'site', 'employed_m12']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if data['female'].isna().any() or not data['female'].isin([0, 1]).all():
        raise ValueError('female must be observed and coded 0 or 1.')
    sample = data.loc[data['female'].eq(1), required].copy()
    if sample.empty or sample.isna().any().any():
        raise ValueError('The female subgroup must be nonempty with complete data.')
    for column in ['treatment', 'employed_m12']:
        if not sample[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be coded 0 or 1.')
    if sample['treatment'].nunique() != 2:
        raise ValueError('The female subgroup must include both treatment arms.')
    design = sm.add_constant(pd.concat([
        sample[['treatment']],
        pd.get_dummies(sample['site'], prefix='site', drop_first=True, dtype=float),
    ], axis=1), has_constant='add').astype(float)
    if (len(sample) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]):
        raise ValueError('Insufficient observations or a rank-deficient design.')
    fit = sm.OLS(sample['employed_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The data do not support finite HC2 inference.')
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
