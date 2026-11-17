#!/usr/bin/env python3
"""Run the single selected Pathways to Work analysis; no exploratory search.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, scipy, and statsmodels.

Fixed analysis: participants with parent == 1; debt_collections_m6 regressed
on assignment, categorical site, and all varying baseline covariates.
Education is categorical. Parent is constant within this subgroup and omitted.
Inference uses HC2 standard errors and a two-sided residual-df Student t test.
The estimate is a probability difference (not percentage points). The p-value
is nominal, without a correction for the exploratory selection on the original
packet. This program fits exactly one outcome model on the supplied dataset.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    'debt_collections_m6 ~ treatment + C(site) + age + female + C(education)'
    ' + household_size + employed_baseline + earnings_baseline'
    ' + months_unemployed_baseline'
)
REQUIRED = (
    'parent', 'debt_collections_m6', 'treatment', 'site', 'age', 'female',
    'education', 'household_size', 'employed_baseline', 'earnings_baseline',
    'months_unemployed_baseline',
)


def score(data_path):
    data = pd.read_csv(data_path)
    missing = sorted(set(REQUIRED) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data['parent'].isna().any() or not data['parent'].isin([0, 1]).all():
        raise ValueError('parent must be observed and coded 0/1.')
    sample = data.loc[data['parent'].eq(1), list(REQUIRED)].copy()
    if sample.empty:
        raise ValueError('No observations with parent == 1.')
    if sample.isna().any().any():
        raise ValueError('Missing values in the selected analysis sample.')
    for column in REQUIRED:
        sample[column] = pd.to_numeric(sample[column], errors='raise')
        if not sample[column].map(math.isfinite).all():
            raise ValueError(f'{column} contains nonfinite values.')
    if set(sample['treatment'].unique()) != {0, 1}:
        raise ValueError('The selected sample must contain both treatment arms, coded 0/1.')
    if not sample['debt_collections_m6'].isin([0, 1]).all():
        raise ValueError('debt_collections_m6 must be coded 0/1.')

    model = smf.ols(FORMULA, data=sample, missing='raise').fit(
        cov_type='HC2', use_t=True,
    )
    if model.df_resid <= 0:
        raise ValueError('Insufficient residual degrees of freedom.')
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('The selected model did not produce finite inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data.')
    try:
        result = score(args.data)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Analysis failed: {error}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
