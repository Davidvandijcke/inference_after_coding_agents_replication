#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis; requires pandas and statsmodels.

Estimand: intention-to-treat effect on month-12 monthly earnings among
participants unemployed at random assignment (employed_baseline == 0).
Model: earnings_m12 ~ treatment + C(site), with an intercept.
Inference: HC2 standard error and two-sided Student t test, using the OLS
residual degrees of freedom. No search or multiplicity adjustment is run here.

Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data_path):
    columns = ['employed_baseline', 'treatment', 'site', 'earnings_m12']
    data = pd.read_csv(data_path, usecols=columns)
    if data.isna().any().any():
        raise ValueError('The analysis columns must be fully observed.')
    for column in ['employed_baseline', 'treatment']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1.')
    data = data.loc[data['employed_baseline'].eq(0)].copy()
    if data.empty or data['treatment'].nunique() != 2:
        raise ValueError('The baseline-unemployed subgroup needs both treatment arms.')
    if not np.isfinite(data['earnings_m12'].to_numpy(dtype=float)).all():
        raise ValueError('earnings_m12 must be finite.')

    model = smf.ols('earnings_m12 ~ treatment + C(site)', data=data,
                    missing='raise')
    if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
        raise ValueError('The treatment effect is not identified with site indicators.')
    result = model.fit(cov_type='HC2', use_t=True)
    answer = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if result.df_resid <= 0 or not all(np.isfinite(v) for v in answer.values()):
        raise ValueError('The dataset does not permit finite HC2 t inference.')
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    answer = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(answer, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')


if __name__ == '__main__':
    main()
