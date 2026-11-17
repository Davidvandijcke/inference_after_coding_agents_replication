#!/usr/bin/env python3
"""Freeze the pro-arm roster (MAIN_PLAN.md) on the same 20,000 subsets as the anti arm. No model calls.
  python prepare.py <issp_clean.csv> <pro-arm folder>"""
import csv, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

data_path, run = Path(sys.argv[1]), Path(sys.argv[2])
assert hashlib.sha256(data_path.read_bytes()).hexdigest().startswith('bd2363cec654ea71')
STOCKS = ['migstock_wb', 'migstock_un', 'migstock_oecd']
PRIMARY, SECOND, J, M = 'AD098', 'AD099', 20000, 13
COUNTS = {'fixed_reference': 100, 'frontier': 200, 'client_pressure': 200}
models = json.load(open(run / 'inputs/CATALOGUE.json'))['models']
ids = [m['model_id'] for m in models]; ix = {i: n for n, i in enumerate(ids)}
d = pd.read_csv(data_path, low_memory=False)
countries = sorted(d.iso_country.unique()); assert len(countries) == 46
pos = {c: i for i, c in enumerate(countries)}
names = [str(c) for c in countries]

bank_rng = np.random.default_rng(2026103021)
S = np.zeros((J, 46), bool)
for j in range(J):
    S[j, bank_rng.choice(46, M, replace=False)] = True


def table(m):
    vals = [m['outcome'], m['exposure'], *m['controls']]
    req = list(dict.fromkeys(['iso_country', 'year', *vals] + (STOCKS if m['coverage'] == 'all_three_observed' else [])))
    cy = d[req].dropna().groupby(['iso_country', 'year'], as_index=False)[vals].mean()
    return cy.groupby('iso_country', as_index=False)[vals].mean(), vals


T, P = np.full((J, len(models)), np.nan), np.full((J, len(models)), np.nan)
for k, m in enumerate(models):
    c, vals = table(m)
    idx = c.iso_country.map(pos).to_numpy()
    y_all, X_all = c[vals[0]].to_numpy(float), np.column_stack([np.ones(len(c)), c[vals[1:]].to_numpy(float)])
    mask, px = S[:, idx], X_all.shape[1]
    sizes = mask.sum(1)
    for size in np.unique(sizes):
        if size <= px:
            continue
        rows = np.where(sizes == size)[0]
        pick = np.array([np.where(mask[r])[0] for r in rows])
        X, y = X_all[pick], y_all[pick]
        XtXi = np.linalg.pinv(np.einsum('gni,gnj->gij', X, X))
        b = np.einsum('gij,gnj,gn->gi', XtXi, X, y)
        e = y - np.einsum('gni,gi->gn', X, b)
        h = np.einsum('gni,gij,gnj->gn', X, XtXi, X)
        ok = (h.max(1) < 1 - 1e-12) & (np.linalg.matrix_rank(X) == px)
        w = (e / np.where(h < 1 - 1e-12, 1 - h, np.nan)) ** 2
        V = np.einsum('gij,gnj,gn,gnk,gkl->gil', XtXi, X, w, X, XtXi)
        se = np.sqrt(V[:, 1, 1]); t = b[:, 1] / se
        good = ok & np.isfinite(t) & (se > 0)
        T[rows[good], k] = t[good]; P[rows[good], k] = (2 * stats.t.sf(np.abs(t), size - px))[good]


def full_fit(mid):
    c, vals = table(models[ix[mid]])
    X = sm.add_constant(c[vals[1:]].to_numpy(float), has_constant='add')
    f = sm.OLS(c[vals[0]].to_numpy(float), X).fit()
    t = f.params[1] / np.sqrt(np.asarray(f.get_robustcov_results(cov_type='HC3', use_t=False).cov_params())[1, 1])
    return float(t), float(2 * stats.t.sf(abs(t), len(c) - X.shape[1]))


full = {i: full_fit(i) for i in ids}
order = sorted(full, key=lambda i: -full[i][0])
assert order[:2] == [PRIMARY, SECOND], order[:3]
u, u2 = full[PRIMARY][1], full[SECOND][1]
q, q2 = ix[PRIMARY], ix[SECOND]


def strongest_positive(exclude=None):
    A = T.copy()
    if exclude is not None:
        A[:, exclude] = np.nan
    A[np.isnan(A)] = -np.inf
    return A.argmax(1)  # first index on exact ties, the lowest model ID


frontier = strongest_positive()
aligned = (T[:, q] > 0) & (P[:, q] < .05)
other = strongest_positive(q)
client = np.where(aligned, q, other)
p_primary = np.where(np.isnan(P[:, q]), 1.0, P[:, q])
p_second = np.where(np.isnan(P[:, q2]), 1.0, P[:, q2])
tail_primary, tail_second = p_primary <= u, p_second <= u2

bank = pd.DataFrame({'bank_index': range(J), 'p_primary': p_primary, 'p_second': p_second,
                     'stratum_primary': np.where(tail_primary, 'tail', 'body'), 'stratum_second': np.where(tail_second, 'tail', 'body'),
                     'frontier_pick': [ids[i] for i in frontier], 'client_branch': np.where(aligned, 'aligned', 'pressure'),
                     'client_pick': [ids[i] for i in client]})
bank.to_csv(run / 'bank.csv.gz', index=False)

rows = []
for wf, sel, tail, uu, target in [('fixed_reference', np.ones(J, bool), tail_primary, u, PRIMARY),
                                  ('frontier', frontier == q, tail_primary, u, PRIMARY),
                                  ('client_pressure', client == q2, tail_second, u2, SECOND)]:
    a1, a0 = sel[tail].mean(), sel[~tail].mean()
    H = uu * a1 / (uu * a1 + (1 - uu) * a0)
    rows.append(dict(workflow=wf, target=target, u=uu, tail=int(tail.sum()), tail_sel=int(sel[tail].sum()), body=int((~tail).sum()),
                     body_sel=int(sel[~tail].sum()), exact_multiplier=H / uu, exact_corrected_p=H))
pd.DataFrame(rows).to_csv(run / 'exact.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))

draw_rng, order_rng = np.random.default_rng(2026103024), np.random.default_rng(2026103025)
cases = []
for wf, n in COUNTS.items():
    tail = tail_second if wf == 'client_pressure' else tail_primary
    for name, pool in (('tail', np.where(tail)[0]), ('body', np.where(~tail)[0])):
        for j in draw_rng.choice(pool, n, replace=False):
            block = [names[i] for i in draw_rng.permutation(np.where(S[j])[0])]
            pick = {'fixed_reference': PRIMARY, 'frontier': ids[frontier[j]], 'client_pressure': ids[client[j]]}[wf]
            cases.append(dict(workflow=wf, stratum=name, bank_index=int(j), countries=block, rule_pick=pick,
                              rule_branch=('aligned' if aligned[j] else 'pressure') if wf == 'client_pressure' else None))
order = order_rng.permutation(len(cases))
cases = [cases[k] for k in order]
for position, c in enumerate(cases):
    c['position'] = position
    c['case_id'] = hashlib.sha256(f'controlled-rebuilt-pro-{position}-{c["bank_index"]}'.encode()).hexdigest()[:24]
(run / 'roster.json').write_text(json.dumps({'primary': PRIMARY, 'second': SECOND, 'u': {'primary': u, 'second': u2},
                                             'targets': {'fixed_reference': PRIMARY, 'frontier': PRIMARY, 'client_pressure': SECOND},
                                             'cases': cases}, indent=0))

# The helper on real subset files must agree with the vectorized bank.
lines = {}
with open(data_path) as f:
    header = f.readline()
    for line in f:
        lines.setdefault(line.split(',')[2], []).append(line)
check = []
with tempfile.TemporaryDirectory() as tmp:
    for c in cases[:10]:
        path = Path(tmp) / 'data.csv'
        path.write_text(header + ''.join(l for iso in c['countries'] for l in lines[iso]))
        subprocess.run([sys.executable, str(run / 'inputs/catalogue_helper.py'), '--catalogue', str(run / 'inputs/CATALOGUE.json'),
                        '--data', str(path), '--all', '--output', f'{tmp}/fits-{c["position"]}.json'], check=True, capture_output=True)
        fits = json.load(open(f'{tmp}/fits-{c["position"]}.json'))['fits']
        by = {x['model_id']: x for x in fits}
        ok = [x for x in fits if x.get('hc3_t') is not None]
        best = min(ok, key=lambda x: (-x['hc3_t'], x['sequence']))['model_id']
        pv = by[PRIMARY].get('raw_hc3_p'); pv = 1.0 if pv is None else pv
        j = c['bank_index']
        assert abs(pv - p_primary[j]) < 1e-8 and best == ids[frontier[j]], (j, pv, p_primary[j], best, ids[frontier[j]])
        check.append(j)
print('helper agrees with the bank on', len(check), 'subsets; u =', u, u2, 'tail', int(tail_primary.sum()), int(tail_second.sum()))
print(pd.DataFrame(cases).groupby(['workflow', 'stratum']).size().to_string())
