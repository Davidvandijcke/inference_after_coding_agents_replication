#!/usr/bin/env python3
"""Rerun the fixed, exploratory Pathways to Work analysis.

Population: baseline education == 2 (high-school diploma or GED).
Outcome: financial security at 12 months, in index points (0–20).
Model: OLS with an intercept, treatment, and categorical site indicators.
Inference: HC2 standard error; two-sided Student t test with OLS residual df.
The reported p-value is nominal, without adjustment for the original search.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, and statsmodels (which depends on scipy).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data_path):
    columns = ['education', 'site', 'treatment', 'fin_security_m12']
    data = pd.read_csv(data_path, usecols=columns)
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors='raise')
    if not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError('Analysis columns must contain finite, observed values.')
    subset = data.loc[data.education.eq(2)].copy()
    if set(subset.treatment.unique()) != {0, 1}:
        raise ValueError('The education=2 subgroup must contain both treatment arms coded 0/1.')

    model = smf.ols('fin_security_m12 ~ treatment + C(site)', data=subset)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('The chosen regression design must have full column rank.')
    if design.shape[0] <= design.shape[1]:
        raise ValueError('The chosen regression needs positive residual degrees of freedom.')
    q, _ = np.linalg.qr(design, mode='reduced')
    if np.any(np.sum(q * q, axis=1) >= 1 - 1e-12):
        raise ValueError('HC2 is undefined for observations with leverage one.')

    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen regression did not produce finite inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
