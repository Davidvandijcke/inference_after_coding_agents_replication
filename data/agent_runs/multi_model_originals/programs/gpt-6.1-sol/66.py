#!/usr/bin/env python3
"""Run the frozen Pathways to Work analysis, with no outcome/subgroup search.

Chosen estimand: intention-to-treat effect on six-month psychological distress
in the full sample. Model: distress_m6 ~ treatment + C(site), including an
intercept, without additional baseline covariates. Inference: HC2 standard
error and a two-sided Student t test using OLS residual degrees of freedom.
The reported p-value is nominal, not adjusted for the original exploration.

Dependencies: numpy, pandas, statsmodels.
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
    required = ['distress_m6', 'treatment', 'site']
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    data = data[required].copy()
    if data.empty or data.isna().any().any():
        raise ValueError('The chosen analysis requires complete, nonempty data.')
    for column in ['distress_m6', 'treatment']:
        data[column] = pd.to_numeric(data[column], errors='raise')
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f'{column} must contain finite numeric values.')
    if set(data['treatment'].unique()) != {0, 1}:
        raise ValueError('treatment must contain both 0 and 1, and no other values.')

    model = smf.ols('distress_m6 ~ treatment + C(site)', data=data,
                    missing='raise')
    if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
        raise ValueError('Treatment and site effects are not separately identifiable.')
    if model.exog.shape[0] <= model.exog.shape[1]:
        raise ValueError('Positive residual degrees of freedom are required.')
    fit = model.fit(cov_type='HC2', use_t=True)
    if np.any(fit.get_influence().hat_matrix_diag >= 1 - 1e-12):
        raise ValueError('HC2 is undefined for observations with unit leverage.')
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen regression produced undefined inference.')
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
