#!/usr/bin/env python3
"""Frozen exploratory hypothesis: treatment effect on debt_collections_m12 at site 1.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, scipy, and statsmodels. No outcome/group/model search is
performed. Estimate is a probability difference (treatment minus control).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    required = ['site', 'treatment', 'debt_collections_m12']
    data = pd.read_csv(data_path, usecols=required)
    for column in required:
        data[column] = pd.to_numeric(data[column], errors='raise')
    if not np.isfinite(data['site']).all():
        raise ValueError('site must contain finite numeric values')
    subset = data.loc[data['site'].eq(1)].copy()
    if subset.empty:
        raise ValueError('The frozen analysis requires participants at site 1')
    for column in ['treatment', 'debt_collections_m12']:
        if not subset[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be observed and binary at site 1')
    counts = subset['treatment'].value_counts()
    if any(counts.get(arm, 0) < 2 for arm in (0, 1)):
        raise ValueError('Site 1 requires at least two observations in each arm for HC2')

    # Within site 1 all site indicators are constant, so an intercept and
    # treatment span exactly the prescribed treatment + site-indicator model.
    design = sm.add_constant(subset[['treatment']].astype(float), has_constant='add')
    model = sm.OLS(subset['debt_collections_m12'].astype(float), design).fit(
        cov_type='HC2', use_t=True
    )
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('The chosen test is undefined for this dataset')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    try:
        result = chosen_analysis(args.data)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
