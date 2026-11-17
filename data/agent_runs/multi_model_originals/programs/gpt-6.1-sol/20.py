#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis, without any model or subgroup search.

Population: months_unemployed_baseline >= 6.
Outcome: debt_collections_m12 (0/1).
Model: outcome ~ treatment + intercept + categorical site fixed effects.
Inference: HC2 standard error, two-sided Student t test, residual degrees
of freedom n - rank(X). Estimates are in probability units, not percent.

Requires numpy, pandas, scipy, and statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data: pd.DataFrame) -> dict[str, float]:
    required = ['months_unemployed_baseline', 'debt_collections_m12',
                'treatment', 'site']
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError('Analysis columns must contain finite numeric values.')
    if not data['treatment'].isin([0, 1]).all():
        raise ValueError('treatment must be coded 0/1.')
    if not data['debt_collections_m12'].isin([0, 1]).all():
        raise ValueError('debt_collections_m12 must be coded 0/1.')

    subgroup = data.loc[data['months_unemployed_baseline'] >= 6].copy()
    if subgroup['treatment'].nunique() != 2:
        raise ValueError('The fixed subgroup must contain both treatment arms.')
    model = smf.ols('debt_collections_m12 ~ treatment + C(site)', data=subgroup)
    design = model.exog
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('Treatment and site effects are not separately estimable.')
    if design.shape[0] <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()) or result['std_error'] <= 0:
        raise ValueError('The fixed analysis did not produce valid inference.')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
