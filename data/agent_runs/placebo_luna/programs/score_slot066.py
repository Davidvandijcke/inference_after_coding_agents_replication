#!/usr/bin/env python3
"""Score the block- and baseline-adjusted year-1 physician/ER visit effect."""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Input CSV with evaluation columns')
    parser.add_argument('--output', required=True, help='Path for JSON result')
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {'physician_or_er_visit_y1', 'physician_or_er_visit_base', 'treatment', 'block'}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f'Missing required columns: {", ".join(sorted(missing))}')

    sample = data[list(required)].dropna()
    model = smf.ols(
        'physician_or_er_visit_y1 ~ treatment + C(block) + physician_or_er_visit_base',
        data=sample,
    ).fit(cov_type='HC2')
    result = {
        'estimate': float(model.params['treatment']),
        'std_error': float(model.bse['treatment']),
        'p_value': float(model.pvalues['treatment']),
    }
    with open(args.output, 'w', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
