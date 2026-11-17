#!/usr/bin/env python3
"""Score the fixed, exploratory Pathways to Work finding.

Analysis: food_insecure_m12 ~ treatment + C(site), restricted to female == 0.
The estimate and its standard error are in probability units (0.01 = 1 pp).
Inference: OLS, HC2 standard error, two-sided t test with residual degrees
of freedom. The p-value is nominal, without adjustment for the original search.

Requires numpy, pandas, and statsmodels. No search or model selection is run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    required = ['female', 'site', 'treatment', 'food_insecure_m12']
    data = pd.read_csv(data_path)
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {", ".join(missing)}')
    for column in required:
        data[column] = pd.to_numeric(data[column], errors='raise')
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f'{column} contains missing or nonfinite values')
    for column in ['female', 'treatment', 'food_insecure_m12']:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0 and 1')
    if not data['site'].isin([1, 2, 3, 4]).all():
        raise ValueError('site must contain only 1, 2, 3, or 4')

    subgroup = data.loc[data['female'].eq(0)].copy()
    if subgroup['treatment'].nunique() != 2:
        raise ValueError('The female=0 subgroup must contain both treatment arms')
    model = smf.ols('food_insecure_m12 ~ treatment + C(site)', data=subgroup)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Treatment and site effects are not separately identified')
    if len(subgroup) <= design.shape[1]:
        raise ValueError('Insufficient residual degrees of freedom')
    inverse = np.linalg.pinv(design)
    leverage = np.einsum('ij,ji->i', design, inverse)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError('HC2 is undefined for observations with unit leverage')

    result = model.fit(cov_type='HC2', use_t=True)
    output = {
        'estimate': float(result.params['treatment']),
        'std_error': float(result.bse['treatment']),
        'p_value': float(result.pvalues['treatment']),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError('The model produced nonfinite inference')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Input CSV')
    parser.add_argument('--output', type=Path, required=True, help='Output JSON')
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    try:
        output = score(args.data)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        parser.exit(2, f'Analysis failed: {error}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
