#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis, without any model selection.

Population: participants with employed_baseline == 0.
Outcome: food_insecure_m12 (in original 0/1 units).
Model: outcome ~ treatment + categorical site indicators, with intercept.
Inference: HC2 standard error; two-sided t-test with OLS residual degrees
of freedom. The p-value is nominal, without a search adjustment.

Dependencies: numpy, pandas, scipy, statsmodels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ['employed_baseline', 'site', 'treatment', 'food_insecure_m12']
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    data = data[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors='raise')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Analysis columns must be observed and finite.')
    for column in ['employed_baseline', 'treatment', 'food_insecure_m12']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be coded 0/1.')
    subgroup = data.loc[data.employed_baseline.eq(0)].copy()
    if subgroup.treatment.nunique() != 2:
        raise ValueError('The fixed subgroup must contain both treatment arms.')
    model = smf.ols('food_insecure_m12 ~ treatment + C(site)', data=subgroup)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Treatment is not separately identifiable from site.')
    if len(subgroup) <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom.')
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('HC2 is undefined for observations with unit leverage.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The requested coefficient or inference is undefined.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
