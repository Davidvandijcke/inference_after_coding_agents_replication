#!/usr/bin/env python3
"""Figure 8 and Table 7 for the 2,983 replays the first declaration review classified whose final program runs:
reported p-values against the original program's on the same subsets (weighted to the reference bank), and the
methodological review of their analyses (unweighted). "Reviewer + audit" passes the failures that a check of the
code and data found to be no error and fails the passes that report a p-value adjusted for undeclared multiple
testing.

Inputs:  data/agent_runs/card_luna/replays.csv.gz, replay-results.csv, replay-programs.jsonl.gz,
         soundness-review.csv.gz; data/intermediate/08_card/card-bank-p.csv;
         data/derived/08_card/choice-pvalues.csv, original.csv
Outputs: paper/figs/card_replay_significance.pdf, paper/tabs/card_replay_review.tex,
         data/derived/08_card/replay-summary.json
"""
import csv
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'data' / 'agent_runs' / 'card_luna'
DERIVED = ROOT / 'data' / 'derived' / '08_card'
AGENT, BENCH, INK, MUTED = '#d6402b', '#2a78d6', '#29333d', '#5f6872'
STEP_COLORS = ('#b4b9be', '#2a78d6', '#eda100', '#1baf7a')
MEASURES = ('p_nonmanager_fte', 'p_full_time_employees', 'p_employee_headcount', 'p_manager_inclusive_fte')
# Ground of a failing verdict, from its reason; the first matching rule wins.
GROUNDS = [('silent_fallback', r'rank|collinear|silently (drop|discard)|perfectly confounded'),
           ('identification', r'parallel[- ]trends|causal identif|identifying restriction'),
           ('outcome', r'managers and assistant|full-time employment share|composition rather'),
           ('closures', r'status2 = 1|asymmetric selection'),
           ('inference', r'HC1|HC3|winsoriz|location-shift|degrees of freedom|state shocks'),
           ('declaration', r'.')]
# What checking each failure against the code and data found, on (ground, reason); the first match wins.
AUDIT = [('disclaimer', r'identification', r'.'),  # objects to a sentence in the declaration
         ('disclaimer', r'inference', r'state shocks|within-state dependence'),
         ('no_effect', r'silent_fallback', r'.'),  # the code path never changes the result on these data
         ('no_effect', r'declaration', r'clos|chain be nonmissing|omits emppt|recomputes chain weights'),
         ('no_effect', r'closures', r'asymmetric'),
         ('choice', r'outcome', r'.'),  # a choice of outcome, sample or degrees of freedom, carried out correctly
         ('choice', r'closures', r'status2 = 1'),
         ('choice', r'inference', r'pooled degrees of freedom'),
         ('confirmed', r'.', r'.')]
ADJUSTMENT = r'p\.adjust|holm|bonferroni|benjamini|\bBH\b'


def betainc(a, b, x):
    """Regularized incomplete beta function (continued fraction, Numerical Recipes betacf)."""
    if x <= 0 or x >= 1:
        return float(x >= 1)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x > (a + 1) / (a + b + 2):
        return 1 - betainc(b, a, 1 - x)
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / d; h = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 / (1 + num * d); c = 1 + num / c; h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return front * h / a


def t_two_sided(t, df):
    return betainc(df / 2, .5, df / (df + t * t))


def strip_comments(code):
    """R code without comments (text after a # outside quotes) and the blank lines they leave, as the reviewer saw it."""
    out = []
    for line in code.split('\n'):
        quote, cut = None, len(line)
        for i, ch in enumerate(line):
            if quote:
                if ch == '\\':
                    continue
                if ch == quote and (i == 0 or line[i - 1] != '\\'):
                    quote = None
            elif ch in ('"', "'", '`'):
                quote = ch
            elif ch == '#':
                cut = i
                break
        kept = line[:cut].rstrip()
        if kept or (line.strip() == '' and out and out[-1] != ''):
            out.append(kept)
    return '\n'.join(out).strip() + '\n'


def audit(review, result, code):
    """Audit outcome of one review: '' for a pass that stands, 'missed' for a pass that reports an adjusted p-value
    for its single declared test, else the outcome of checking the failure."""
    reason = review['review_reason']
    if review['review_pass'] == 'true':
        p, implied = float(result['p_value']), t_two_sided(abs(float(result['estimate']) / float(result['se'])), 140)
        return 'missed' if re.search(ADJUSTMENT, strip_comments(code), re.I) and p > 1.5 * implied else ''
    ground = next(g for g, pat in GROUNDS if re.search(pat, reason))
    return next(a for a, g, pat in AUDIT if re.search(g, ground) and re.search(pat, reason))


def load():
    with gzip.open(RUNS / 'replays.csv.gz', 'rt', encoding='utf-8') as f:
        replays = {r['session_id']: r for r in csv.DictReader(f)}
    results = {r['session_id']: r for r in csv.DictReader(open(RUNS / 'replay-results.csv', encoding='utf-8'))
               if r['p_value'] not in ('', 'NA')}
    with gzip.open(RUNS / 'replay-programs.jsonl.gz', 'rt', encoding='utf-8') as f:
        code = {r['session_id']: r['analysis_R'] for r in map(json.loads, f)}
    with gzip.open(RUNS / 'soundness-review.csv.gz', 'rt', encoding='utf-8') as f:
        review = {r['session_id']: r for r in csv.DictReader(f)}
    choice = {r['session_id']: r for r in csv.DictReader(open(DERIVED / 'choice-pvalues.csv', encoding='utf-8'))}
    ids = sorted(k for k in results if replays[k]['first_review_status'] == 'classified')
    u = next(float(r['value']) for r in csv.DictReader(open(DERIVED / 'original.csv', encoding='utf-8')) if r['name'] == 'u')
    bank = np.array([float(r['p_value']) for r in
                     csv.DictReader(open(ROOT / 'data/intermediate/08_card/card-bank-p.csv', encoding='utf-8'))])
    tail = np.array([replays[k]['stratum'] == 'tail' for k in ids]); share = float((bank <= u).mean())
    # each stratum weighted by its share of the bank
    weight = np.where(tail, share / tail.sum(), (1 - share) / (~tail).sum())
    num = lambda rows, key: np.array([float(rows[k][key]) for k in ids])
    outcome = [audit(review[k], results[k], code[k]) for k in ids]
    return {'ids': ids, 'tail': tail, 'weight': weight, 'p0': num(replays, 'bank_p'), 'p1': num(results, 'p_value'),
            'measure': num(choice, 'p_nj_additive_chain'), 's_rule': num(choice, 's_rule'),
            'best': np.array([min(float(choice[k][m]) for m in MEASURES) for k in ids]),
            'passed': np.array([review[k]['review_pass'] == 'true' for k in ids]),
            'audited': np.array([a not in ('confirmed', 'missed') for a in outcome]), 'outcome': outcome,
            'second_reader_passed': np.array([review[k]['audit_major_error'] == 'false' for k in ids])}


def summarize(x):
    share = lambda v: float(x['weight'][v < .05].sum())
    groups = {'all': np.ones(len(x['ids']), bool), 'tail': x['tail'], 'body': ~x['tail']}
    return {'replays': len(x['ids']),
            'significant_share': {'original': share(x['p0']), 'measure': share(x['measure']), 'sample_rule': share(x['s_rule']),
                                  'replay': share(x['p1']), 'best_of_four': share(x['best'])},
            'review': {k: {'replays': int(m.sum()), 'pass': int((m & x['passed']).sum()),
                           'pass_audited': int((m & x['audited']).sum()),
                           'pass_second_reader': int((m & x['second_reader_passed']).sum())} for k, m in groups.items()},
            'audit': {a: x['outcome'].count(a) for a in ('disclaimer', 'no_effect', 'choice', 'confirmed', 'missed')}}


def review_table(s, path):
    g = s['review']
    pct = lambda k, f: f"{100 * g[k][f] / g[k]['replays']:.1f}"
    lines = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule', r'& & \multicolumn{2}{c}{Pass (\%)} \\', r'\cmidrule(l){3-4}',
             r' & Replays & Reviewer & Reviewer + audit \\', r'\midrule']
    lines += [f"{name} & {g[k]['replays']:,} & {pct(k, 'pass')} & {pct(k, 'pass_audited')} \\\\"
              for k, name in (('all', 'All replays'), ('tail', 'Tail datasets'), ('body', 'Body datasets'))]
    lines += [r'\bottomrule', r'\end{tabular}']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def figure(x, s, path):
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.edgecolor': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
    w = s['significant_share']
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={'width_ratios': [1, 1.3]})
    fig.subplots_adjust(left=.075, right=.98, top=.9, bottom=.2, wspace=.25)
    a = ax[1]; prev = 0
    vals = [100 * w[k] for k in ('original', 'measure', 'sample_rule', 'replay')]
    names = ['Original\nanalysis', "+ replay's\noutcome", '+ its\nsample rule', '+ its full\nspecification']
    b = 100 * w['best_of_four']
    for j, (v, col) in enumerate(zip(vals, STEP_COLORS)):
        a.bar(j, v - prev, bottom=prev, color=col, width=.62)
        if b - 4 < v < b:  # keep the label clear of the dashed line
            a.text(j, prev - 1, f'{v:.0f}%', ha='center', va='top', fontsize=7.2)
        else:
            a.text(j, v + 1, f'{v:.0f}%', ha='center', fontsize=7.2)
        if 0 < j < 3:
            a.plot([j + .31, j + .69], [v, v], color='#999999', lw=.6)
        prev = v
    a.axhline(b, color=INK, lw=.8, ls='--'); a.text(-.4, b + .8, 'best of four\noutcome measures', ha='left', va='bottom', fontsize=6.2)
    a.set_xticks(range(4), names, fontsize=6.3); a.set_ylim(0, 40); a.set_ylabel('Subsamples with $p$ < 0.05 (%)')
    a.set_title('(b) Significant at 5%', loc='left')
    a = ax[0]; bins = np.arange(0, .1501, .005)
    a.hist(x['p1'], bins=bins, weights=100 * x['weight'], color=AGENT, alpha=.75, label='Replay agents')
    a.hist(x['best'], bins=bins, weights=100 * x['weight'], histtype='step', color=BENCH, lw=1, label='Best of four measures')
    a.hist(x['p0'], bins=bins, weights=100 * x['weight'], histtype='step', color=MUTED, lw=1, ls='--', label='Original analysis')
    a.axvline(.05, color=INK, lw=.7); a.set_xlim(0, .15); a.set_xticks([0, .05, .1, .15], ['0', '0.05', '0.10', '0.15'])
    a.set_xlabel('Reported $p$-value'); a.set_ylabel('Subsamples (%)')
    a.set_title('(a) Reported $p$-values near 0.05', loc='left')
    a.legend(loc='upper right', frameon=False, fontsize=6.6, handlelength=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)


def main():
    x = load()
    s = summarize(x)
    figure(x, s, ROOT / 'paper/figs/card_replay_significance.pdf')
    review_table(s, ROOT / 'paper/tabs/card_replay_review.tex')
    (DERIVED / 'replay-summary.json').write_text(json.dumps(s, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
