#!/usr/bin/env python3
"""Fixed exploratory finding: 12-month financial security for age < 35.

Fit OLS with an intercept, treatment, and site indicators in this subgroup.
Report the treatment coefficient, HC2 standard error, and two-sided t-test
p-value using residual degrees of freedom. The p-value is nominal; this
program does not redo the exploratory search or adjust for its selection.

Dependencies: numpy, pandas, scipy (via statsmodels), statsmodels.
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
    required = ['age', 'site', 'treatment', 'fin_security_m12']
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if data[required].isna().any().any():
        raise ValueError('Required columns must have no missing values.')
    numeric = data[['age', 'treatment', 'fin_security_m12']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('Age, treatment, and outcome must be finite numbers.')
    if not data.treatment.isin([0, 1]).all():
        raise ValueError('Treatment must be coded 0 or 1.')

    # This outcome, follow-up, subgroup, and model are fixed in advance of
    # reading any new dataset. No outcome comparisons or model selection.
    subgroup = data.loc[data.age.lt(35)]
    if subgroup.empty or subgroup.treatment.nunique() != 2:
        raise ValueError('The age < 35 subgroup must contain both treatment arms.')
    X = pd.concat([
        subgroup[['treatment']],
        pd.get_dummies(subgroup.site, prefix='site', drop_first=True, dtype=float),
    ], axis=1)
    X = sm.add_constant(X, has_constant='add').astype(float)
    if len(X) <= X.shape[1] or np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError('The fixed regression requires a full-rank design and positive residual degrees of freedom.')
    design = X.to_numpy()
    leverage = np.einsum('ij,ji->i', design, np.linalg.pinv(design))
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('HC2 inference is undefined for an observation with unit leverage.')
    model = sm.OLS(subgroup.fin_security_m12, X).fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The fixed regression did not produce finite inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
