#!/usr/bin/env python3
"""Fixed Pathways to Work analysis, selected after exploring the original packet.

Restrict to education == 3 (some college). Regress training_m6 on an
intercept, treatment, and site indicators, with no other covariates.
Return the treatment coefficient in probability units, its HC2 standard
error, and the two-sided t-test p-value using OLS residual degrees of freedom.
The reported p-value is nominal: it does not adjust for the original search.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, scipy, and statsmodels; reads no other files.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    columns = ['education', 'training_m6', 'treatment', 'site']
    data = pd.read_csv(data_path, usecols=columns)
    data = data.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Analysis columns must be complete and finite.')
    subgroup = data.loc[data.education.eq(3)].copy()
    if subgroup.empty or set(subgroup.treatment.unique()) != {0, 1}:
        raise ValueError('The education=3 subgroup must contain both treatment arms.')
    if not subgroup.training_m6.isin([0, 1]).all():
        raise ValueError('training_m6 must be binary.')

    model = smf.ols('training_m6 ~ treatment + C(site)', data=subgroup,
                    missing='raise')
    design = model.exog
    if (len(subgroup) <= design.shape[1] or
            np.linalg.matrix_rank(design) < design.shape[1]):
        raise ValueError('Insufficient independent observations for the fixed model.')
    fit = model.fit(cov_type='HC2', use_t=True)
    result = {
        'estimate': float(fit.params['treatment']),
        'std_error': float(fit.bse['treatment']),
        'p_value': float(fit.pvalues['treatment']),
    }
    if not all(np.isfinite(v) for v in result.values()) or result['std_error'] <= 0:
        raise ValueError('The fixed model did not produce finite, nondegenerate inference.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error('--output must differ from --data')
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
