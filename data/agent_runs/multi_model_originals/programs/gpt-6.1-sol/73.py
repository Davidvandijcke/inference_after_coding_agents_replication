#!/usr/bin/env python3
"""Frozen analysis: six-month financial security among parent == 0.

Fit OLS fin_security_m6 ~ treatment + categorical site indicators within this
baseline-defined subgroup. Report the treatment coefficient, HC2 standard
error, and two-sided Student t p-value with residual degrees of freedom.
Requires pandas, numpy, and statsmodels. No model or subgroup search is run.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    columns = ['parent', 'treatment', 'site', 'fin_security_m6']
    data = pd.read_csv(data_path, usecols=columns)
    data = data.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Analysis columns must contain finite, observed values.')
    if not data['parent'].isin([0, 1]).all():
        raise ValueError('parent must be coded 0 or 1.')
    if not data['treatment'].isin([0, 1]).all():
        raise ValueError('treatment must be coded 0 or 1.')
    selected = data.loc[data['parent'] == 0].copy()
    if selected['treatment'].nunique() != 2:
        raise ValueError('The parent == 0 subgroup must contain both assignment arms.')

    fit = smf.ols('fin_security_m6 ~ treatment + C(site)', data=selected).fit(
        cov_type='HC2', use_t=True)
    if fit.model.rank != fit.model.exog.shape[1] or fit.df_resid <= 0:
        raise ValueError('The selected regression must be identified with positive residual df.')
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()) or result['std_error'] <= 0:
        raise ValueError('HC2 inference is undefined for this dataset.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
