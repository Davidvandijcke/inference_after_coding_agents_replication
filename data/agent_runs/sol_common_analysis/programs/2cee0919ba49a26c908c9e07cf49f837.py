"""Estimate the 12-month employment ITT using OLS, site effects, and HC2.

Dependencies: pandas, numpy, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    columns = ['treatment', 'site', 'employed_m12']
    selected = data[columns]
    if selected.isna().any().any():
        raise ValueError('The selected analysis requires complete observations.')
    for column in ['treatment', 'employed_m12']:
        if not selected[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1.')

    sites = pd.get_dummies(selected['site'], prefix='site', drop_first=True,
                           dtype=float)
    design = sm.add_constant(
        pd.concat([selected[['treatment']].astype(float), sites], axis=1),
        has_constant='add',
    )
    if (len(design) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]):
        raise ValueError('Insufficient observations or unidentified treatment effect.')

    model = sm.OLS(selected['employed_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True,
    )
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    Path(args.output).write_text(json.dumps(result, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
