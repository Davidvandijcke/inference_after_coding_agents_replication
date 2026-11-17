#!/usr/bin/env python3
"""Rerun the fixed, selected Pathways to Work analysis; perform no search.

Outcome: fin_security_m12. Subgroup: parent == 1 at baseline.
Model: outcome ~ treatment + C(site), with an intercept.
Inference: HC2 standard error and two-sided Student t test, residual df.
Dependencies: pandas, numpy, statsmodels (and its scipy dependency).

Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data_path):
    data = pd.read_csv(data_path)
    required = ['parent', 'treatment', 'site', 'fin_security_m12']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if data[required].isna().any().any():
        raise ValueError('Required analysis variables must be observed.')
    for column in ['parent', 'treatment']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be coded 0 or 1.')
    parents = data.loc[data['parent'].eq(1), required].copy()
    if parents['treatment'].nunique() != 2:
        raise ValueError('The parent subgroup must contain both treatment arms.')
    if not np.isfinite(parents['fin_security_m12'].to_numpy(dtype=float)).all():
        raise ValueError('The outcome must be finite.')
    model = smf.ols('fin_security_m12 ~ treatment + C(site)', data=parents)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('The selected regression design must have full rank.')
    if len(parents) <= design.shape[1]:
        raise ValueError('The regression needs positive residual degrees of freedom.')
    leverage = np.einsum('ij,ji->i', design, np.linalg.pinv(design))
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('HC2 inference requires observation leverage below one.')
    result = model.fit(cov_type='HC2', use_t=True)
    output = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError('The selected regression produced nonfinite results.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    output = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
