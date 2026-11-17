#!/usr/bin/env python3
"""Run the frozen Pathways to Work analysis, without searching.

Estimand: assignment effect on fin_security_m12 among participants whose
baseline education is 3 (some college) or 4 (bachelor's degree or higher).
Model: OLS with intercept, treatment, and categorical site fixed effects.
Inference: HC2 standard error; two-sided Student t test, residual degrees
of freedom n - rank(X). The reported p-value is nominal, not search-adjusted.

Requires numpy, pandas, scipy, and statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    required = ['education', 'site', 'treatment', 'fin_security_m12']
    data = pd.read_csv(data_path)
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    data = data[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Required columns must contain finite, observed values.')
    if not data.education.isin([1, 2, 3, 4]).all():
        raise ValueError('Education must use codebook categories 1 through 4.')
    if not data.treatment.isin([0, 1]).all():
        raise ValueError('Treatment must be coded 0 or 1.')

    # This subgroup, endpoint, and specification are fixed; no alternatives run.
    subgroup = data.loc[data.education.isin([3, 4])].reset_index(drop=True)
    if subgroup.treatment.nunique() != 2:
        raise ValueError('The chosen subgroup must contain both treatment arms.')
    sites = pd.get_dummies(subgroup.site, prefix='site', drop_first=True,
                           dtype=float)
    design = sm.add_constant(
        pd.concat([subgroup[['treatment']].astype(float), sites], axis=1),
        has_constant='add',
    )
    matrix = design.to_numpy(dtype=float)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError('The chosen regression design is not full rank.')
    if len(subgroup) <= matrix.shape[1]:
        raise ValueError('The chosen regression needs positive residual degrees of freedom.')
    leverage = np.sum(np.linalg.qr(matrix, mode='reduced')[0] ** 2, axis=1)
    if np.any(1 - leverage <= 1e-12):
        raise ValueError('HC2 is undefined for an observation with unit leverage.')

    fit = sm.OLS(subgroup.fin_security_m12, design).fit(
        cov_type='HC2', use_t=True,
    )
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('The chosen regression produced nonfinite results.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    try:
        result = analyze(args.data)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
