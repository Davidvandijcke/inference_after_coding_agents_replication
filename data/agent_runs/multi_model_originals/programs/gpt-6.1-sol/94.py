#!/usr/bin/env python3
"""Estimate the fixed, exploratory Pathways to Work contrast.

Chosen analysis: training_m6 on treatment and categorical site indicators,
restricted to baseline education codes 1 or 2. OLS with HC2 standard errors
and a two-sided Student t test using residual degrees of freedom.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels (see requirements.txt).
This program performs no outcome, subgroup, or model selection. Its reported
p value is nominal: it does not adjust for the original exploratory search.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data: pd.DataFrame) -> dict[str, float]:
    required = ['treatment', 'site', 'education', 'training_m6']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data[required].isna().any().any():
        raise ValueError('Required analysis columns must be fully observed.')
    for column, allowed in [('treatment', [0, 1]), ('training_m6', [0, 1]),
                            ('education', [1, 2, 3, 4])]:
        if not data[column].isin(allowed).all():
            raise ValueError(f'{column} must contain only {allowed}.')

    sample = data.loc[data.education.isin([1, 2]), required].copy()
    if sample.treatment.nunique() != 2:
        raise ValueError('The chosen subgroup must contain both treatment arms.')
    model = smf.ols('training_m6 ~ treatment + C(site)', data=sample)
    design = model.exog
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('Treatment is not identified separately from site.')
    if design.shape[0] <= design.shape[1]:
        raise ValueError('Positive residual degrees of freedom are required.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen regression has undefined inference.')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    try:
        result = chosen_analysis(pd.read_csv(args.data))
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
