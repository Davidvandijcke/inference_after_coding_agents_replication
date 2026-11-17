#!/usr/bin/env python3
"""Rerun the single chosen analysis: effect of treatment on 6-month financial
security (fin_security_m6), full sample, OLS with site indicators, HC2 SEs,
two-sided t-test on the treatment coefficient. No search is repeated."""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    m = smf.ols('fin_security_m6 ~ treatment + C(site)', d).fit(cov_type='HC2')
    out = {'estimate': float(m.params['treatment']),
           'std_error': float(m.bse['treatment']),
           'p_value': float(m.pvalues['treatment'])}
    with open(a.output, 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()
