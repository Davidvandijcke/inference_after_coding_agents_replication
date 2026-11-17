#!/usr/bin/env python3
"""Figure 9: tests stated by GPT-6.1-Sol's Card-Krueger reports, and the share of its replays outside the original's family.

A replay is outside the family when the outcome, controls and sample rule of its declaration, coded by keyword, form
none of the original's 26 cells. The lenient count treats extra or partial controls as adjusted, the strict count not.

Inputs:  data/agent_runs/card_sol/ (module 09), data/agent_runs/card_placebo_sol/blind_reads.csv.gz,
         data/derived/10_self_correction/card_sol_interval.csv
Outputs: paper/figs/self_correction_family.pdf, data/derived/10_self_correction/card_family_coding.csv, numbers_card.json
"""
import csv
import gzip
import json
import math
import os
import re
import statistics
from pathlib import Path

os.environ['SOURCE_DATE_EPOCH'] = '0'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'data' / 'agent_runs'
DERIVED = ROOT / 'data' / 'derived' / '10_self_correction'
BLUE, WINE = '#335e8c', '#882255'
NEG = re.compile(r'\b(no|not|without|rather than|instead of|absence of|neither|nor|never|unadjusted for|free of|does not|did not)\b')


def low(c, *fields):
    return ' '.join(c.get(f) or '' for f in fields).lower()


def positive(pattern, t):
    """A match of pattern not preceded within 45 characters by a negation."""
    return any(not NEG.search(t[max(0, m.start() - 45):m.start()]) for m in re.finditer(pattern, t))


def outcome(c):
    t = low(c, 'outcome') or low(c, 'model')
    if re.search(r'\bwage|price|hours|opening|meal|full[- ]price|bonus', t) and not re.search(r'fte|full[- ]time|employ|staff|worker', t):
        return 'other_outcome'
    if re.search(r'symmetric', t):
        return 'symmetric_fte'
    if positive(r'\blog\b|logarithm|log[- ](change|difference|employment|fte|point)|asinh|arcsinh|percent(age)? change|proportion(al|ate)? change|relative change|growth rate|arc[- ]?elasticity', t):
        return 'growth_other'
    if re.search(r'empft2 ?[-−–] ?empft\b|number of full[- ]time (nonmanager |non-manager )?employees|full[- ]time (employee |nonmanager |worker )?counts?\b|full[- ]time employees? (only|alone)', t) \
            and not re.search(r'0\.5|half|emppt|part[- ]time[^.]{0,20}(plus|and|\+|weight)', t):
        return 'full_time'
    fte = positive(r'\bfte|full[- ]time[- ]equivalent|0\.5 ?[×x*·] ?(part|emppt)|half[- ](weight|of|the|a)|part[- ]time[^.]{0,30}(0\.5|half)', t)
    excl = re.search(r'exclud\w*[^.]{0,40}manager|manager[^.]{0,40}exclud|non-?manag|without (the )?managers|omit\w*[^.]{0,25}manager|manager[^.]{0,40}(weight|count)\w*[^.]{0,12}(zero|\b0\b)|manager[^.]{0,30}(not|are not|were not) (counted|included)', t)
    incl = re.search(r'includ\w*[^.]{0,30}manager|manager[^.]{0,40}includ|plus (the )?(number of )?managers|\+ ?(nmgrs|managers)|manager[^.]{0,30}(weight|count)\w*[^.]{0,12}\b(1|one)\b|total employment|all employees|employees of all types', t)
    part = re.search(r'\bpart[- ]time', t)
    full = re.search(r'\bfull[- ]time (workers|employees|employment|staff|count|headcount|nonmanag|non-manag)', t)
    if fte:
        if excl and not incl:
            return 'worker_fte'
        if incl and not excl:
            return 'fte'
        if incl and excl:
            return 'worker_fte' if re.search(r'exclud\w*[^.]{0,40}manager|non-?manag', t) else 'fte'
        return 'fte_managers_unstated'
    if re.search(r'headcount|head count|unweighted|raw count|number of (employees|workers|staff)|total (number of )?(employees|workers|staff)|sum of full[- ]time and part[- ]time|full[- ]time[^.]{0,40}(plus|and|\+)[^.]{0,25}part[- ]time[^.]{0,40}(plus|and|\+)?[^.]{0,20}manager', t) and not (part and not full):
        return 'headcount'
    if part and full:
        if re.search(r'(exclud|omit|without|not includ)\w*[^.]{0,40}part[- ]time|part[- ]time[^.]{0,30}(excluded|omitted|not counted)', t):
            return 'full_time'
        if re.search(r'(exclud|omit|without|not includ)\w*[^.]{0,40}full[- ]time|full[- ]time[^.]{0,30}(excluded|omitted|not counted)', t):
            return 'part_time'
        return 'headcount'
    if part:
        return 'part_time'
    if full:
        return 'full_time'
    return 'other_outcome'


def controls(c):
    t = low(c, 'model')
    neg = re.search(r'(no|without|not)[^.]{0,40}(chain|ownership|covariate|control|regressor|conditioning)', t)
    chain = re.search(r'i\(chain|chain[_ ]?(i|indicator|dumm|fixed|effect|categor|specific)|[γβδθ]\w*[^.]{0,15}chain|\+[^.]{0,25}chain|adjust\w*[^.]{0,25}chain|control\w*[^.]{0,25}chain|conditional on[^.]{0,40}chain|chain[^.]{0,20}(indicator|dumm|fixed)', t)
    own = re.search(r'company[- _]?own|co_owned|ownership', t)
    other = positive(r'(control|adjust|covariate|regressor|conditioning on)\w*[^.]{0,50}(baseline (employment|fte|level|wage|staffing|outcome)|starting wage|wage gap|region|southern|central|shore|hours of operation)'
                     r'|[|,]\s*[a-z]*_?(pre|i0|0|before|baseline)(_i)?\b[^.]{0,10}[\]=]|[βγδθλ]\w*\s*\*?\s*\(?[a-z]*_?(pre|i0|before|baseline)(_i)?\b|\+\s*[βγδθλ]?\w*\s*\*?\s*(y|fte|e)_?(pre|0|i0|before)(_i)?\b'
                     r'|(lagged|pre-period|baseline|initial) (outcome|level|employment|fte)[^.]{0,30}(covariate|control|regressor|included|enters)|control\w*[^.]{0,30}(baseline|pre-period|initial) (employment|fte|level|outcome)', t)
    if neg and not (chain and re.search(r'\+[^.]{0,25}chain|i\(chain|chain[_ ]?(i|indicator|dumm|fixed)', t)):
        chain = own = None
    if other and not (chain or own):
        return 'other_controls'
    if other:
        return 'chain_ownership_plus_other'
    if chain and own:
        return 'adjusted'
    if chain or own:
        return 'chain_or_ownership_only'
    return 'unadjusted'


def sample_rule(c):
    s = low(c, 'sample_restrictions')
    closure = low(c, 'sample_restrictions', 'outcome', 'model')
    other = positive(r'exclud\w*[^.]{0,30}(burger king|kfc|roy rogers|wendy|one chain|a chain|the \w+ chain)|only (burger king|kfc|roy rogers|wendy)|trimm(ed|ing) (the|stores|observations|at|changes)|winsoriz(ed|ing) (the|at|changes|employment)'
                     r'|(exclud|remov|dropp)\w* (the |\d+ |all )?(influential|extreme|outlier|outlying)|wage (gap|above|below|at least|of at)|starting wage|(new jersey|pennsylvania) (only|stores only)|single state', s)
    status1 = re.search(r'status2? ?(= ?|of |equal to |equals |is )?1\b(?![^.]{0,12}(or|and|,) ?3)', s)
    excl_closed = re.search(r'exclud\w*[^.]{0,60}(permanent\w* clos|closed stores|closures|stores that closed)|drop\w*[^.]{0,40}(closed|closures)|(open|operating|in business)[^.]{0,20}(in|at|during) both|respondents? only|answered the (follow-up|second)', s)
    perm_zero_coding = re.search(r'permanent\w*[^.]{0,80}(zero|\b0\b)|(zero|\b0\b)[^.]{0,60}permanent|status\w*[^.]{0,25}\b1\b[^.]{0,35}\b(or|and)\b[^.]{0,6}\b3\b|status\w*[^.]{0,25}\b3\b[^.]{0,35}\b(or|and)\b[^.]{0,6}\b1\b|\{1, ?3\}|closures? (at|as|to|coded) zero|closed stores[^.]{0,40}zero|permanent\w* clos\w*[^.]{0,30}(retain|remain|eligible|included|kept)', closure)
    temp_zero = re.search(r'(temporar\w+|all|any|documented|every) (clos\w+|closures)[^.]{0,60}(assign|set|cod|treat|record|count|receiv|fix|impute|given)\w*[^.]{0,25}(zero|\b0\b)'
                          r'|(assign|set|cod|treat|impute|give)\w*[^.]{0,40}(zero|\b0\b)[^.]{0,50}(temporar\w+|all|any|documented|every) (clos\w+|closures)'
                          r'|status2? ?(in|of|=)? ?\{?2,? ?4,? ?(and |or )?5\}?[^.]{0,60}(zero|\b0\b)|status2? (codes? )?(in |of )?\{?1,? ?2,? ?3,? ?4,? ?(and |or )?5\}?|clos\w+ (for any reason|of any kind|of either kind)[^.]{0,40}zero', closure)
    temp_excluded = re.search(r'(exclud|drop|omit|remov)\w*[^.]{0,80}(temporar|renovation|highway|fire|status2? ?(of |in |= ?)?\{?2,? ?4,? ?(and |or )?5)', s)
    other = other or positive(r'(shore|region|southern|central)[^.]{0,30}(equal to|= ?|indicator|only|restrict)|restrict\w*[^.]{0,30}(shore|region|southern|central)', s)
    complete = re.search(r'complete[^.]{0,80}(both|two) (waves|periods|survey)|(both|two) (waves|periods|surveys)[^.]{0,80}(complete|non-?missing|observed|recorded)|non-?missing[^.]{0,80}(both|follow-up|second|wave 2|empft2|emppt2)|complete[- ]case', s)
    if other:
        return 'other_rule'
    if temp_zero and not temp_excluded:
        return 'all_closed_zero'
    if (status1 and not perm_zero_coding) or (excl_closed and not perm_zero_coding and not temp_zero):
        return 'respondents_only'
    if perm_zero_coding or (complete and not status1):
        return 'permanent_zero'
    if status1:
        return 'respondents_only'
    return 'unclear'


def original():
    o = json.load(open(RUNS / 'card_sol' / 'original.json', encoding='utf-8'))
    analyses = list(csv.DictReader(open(RUNS / 'card_sol' / 'original' / 'all_analyses.csv', encoding='utf-8')))
    family = {(a['outcome'], 'adjusted' if a['adjusted'] == 'TRUE' else 'unadjusted', a['closure']) for a in analyses}
    assert len(family) == len(analyses) == 26
    # Holm-adjusted p-values over the 26 tests
    p = sorted(float(a['p_value']) for a in analyses)
    holm, running = [], 0.0
    for i, x in enumerate(p):
        running = max(running, min(1.0, (len(p) - i) * x))
        holm.append(running)
    selected = next(a for a in analyses if a['selected'] == 'TRUE')
    u = float(o['p_value'])
    assert abs(float(selected['p_value']) - u) < 1e-12
    return dict(u=u, tests=len(analyses), family=family, holm=holm[p.index(float(selected['p_value']))])


def family_coding(family):
    rows = []
    with gzip.open(RUNS / 'card_sol' / 'replays.csv.gz', 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['review_label'] not in ('match', 'nonmatch'):
                continue
            c = {k: r[f'commit_{k}'] for k in ('model', 'outcome', 'sample_restrictions')}
            o, a, s = outcome(c), controls(c), sample_rule(c)
            lenient = 'adjusted' if a in ('chain_ownership_plus_other', 'chain_or_ownership_only') else a
            rows.append(dict(session_id=r['session_id'], stratum=r['stratum'], review_label=r['review_label'], outcome=o, controls=a,
                             sample_rule=s, inside_strict=(o, a, s) in family, inside_lenient=(o, lenient, s) in family))
    return sorted(rows, key=lambda r: r['session_id'])


def figure(counts, orig, interval, path):
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#777777', 'axes.linewidth': .6, 'pdf.fonttype': 42})
    edges = list(range(0, 61, 5)) + [10 ** 6]
    labels = [f'{lo + 1}–{hi}' for lo, hi in zip(edges[:-2], edges[1:-1])] + ['61+']
    heights = [sum(lo < k <= hi for k in counts) for lo, hi in zip(edges[:-1], edges[1:])]
    fig, a = plt.subplots(figsize=(5.2, 2.9))
    x = list(range(len(heights)))
    a.bar(x, heights, width=0.82, color=BLUE, lw=0, zorder=2)
    shaded = next(i for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])) if lo < orig['tests'] <= hi)
    a.bar([shaded], [heights[shaded]], width=0.82, color=WINE, lw=0, zorder=3)
    a.set_xticks(x)
    a.set_xticklabels(labels, rotation=60, ha='right')
    a.set_xlabel('Tests the replay report states')
    a.set_ylabel('Replay reports')
    a.set_ylim(0, 1.36 * max(heights))
    a.grid(axis='y', color='.9', lw=.7, zorder=0)
    a.text(.98, .97, f"original run: {orig['tests']} tests, Holm $p$ = {orig['holm']:.2f}\n"
           f"replay correction: $p$ = {interval['p_corrected']:.3f} [{interval['lower95']:.3f}, {interval['upper95']:.3f}]\n"
           f"{len(counts)} replay reports: {min(counts)} to {max(counts)}, median {statistics.median(counts):.0f}", transform=a.transAxes, ha='right', va='top', fontsize=7.5)
    fig.subplots_adjust(bottom=.28, left=.12, right=.98, top=.96)
    fig.savefig(path, metadata={'Title': "Tests stated by GPT-6.1-Sol's Card replay reports", 'CreationDate': None})
    plt.close(fig)


def main():
    orig = original()
    rows = family_coding(orig['family'])
    with open(DERIVED / 'card_family_coding.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    reads = [r for r in csv.DictReader(open(RUNS / 'card_sol' / 'report_reads.csv', encoding='utf-8')) if r['read'] == '1']
    counts = [int(r['tests_stated']) for r in reads if r['tests_stated']]
    with gzip.open(RUNS / 'card_placebo_sol' / 'blind_reads.csv.gz', 'rt', encoding='utf-8') as f:
        full = [float(r['tests_stated']) for r in csv.DictReader(f) if r['tests_stated'] not in ('', 'NA')]
    quartiles = statistics.quantiles(full, n=4)  # Python's default (exclusive) method
    interval = {k: float(v) for k, v in next(csv.DictReader(open(DERIVED / 'card_sol_interval.csv', encoding='utf-8'))).items()}
    assert abs(interval['u'] - orig['u']) < 1e-15
    figure(counts, orig, interval, ROOT / 'paper' / 'figs' / 'self_correction_family.pdf')
    n = len(rows)
    strict, lenient = sum(not r['inside_strict'] for r in rows), sum(not r['inside_lenient'] for r in rows)
    cells = {(r['outcome'], r['controls'], r['sample_rule']) for r in rows}
    numbers = [
        ('card_sol_original_tests', r'the Sol agent explicitly counts the 26 specifications it ran', orig['tests']),
        ('card_sol_original_p', r'increases the original $p$-value of $0.027$', orig['u']),
        ('card_sol_original_holm', r'applies a Holm correction that increases the original $p$-value of $0.027$ to $0.70$', orig['holm']),
        ('card_sol_replay_tests_range', r'between 6 and 194 on the subsampled datasets', [min(counts), max(counts)]),
        ('card_placebo_tests_range', r'between 12 and 1,252 in 200 placebo runs on the full data', [min(full), max(full)]),
        ('card_placebo_runs', r'in 200 placebo runs on the full data, where we permute the treatment assignment', len(full)),
        ('card_sol_outside_family', r'25 to 36\% of the Sol replays select analyses different from the 26', [100 * lenient / n, 100 * strict / n]),
        ('card_sol_distinct_analyses', r'129 distinct analyses across our replays', len(cells)),
        ('card_sol_bonferroni_to_one', r'a Bonferroni correction over 38 or more analyses already sets the original $p$-value of $0.027$ to $1$',
         math.ceil(1 / orig['u'])),
        ('card_sol_replay_p', r'our corrected $p$-value for the Sol agent, $0.053$', interval['p_corrected']),
        ('card_sol_reads', r'through blind reads of 831 reports', len(reads)),
        ('card_sol_reads_with_count', r'823 of which state a count', len(counts)),
        ('card_sol_original_tests_note', r'Shaded bin: contains 26, the count stated by the original run on all 410 stores', orig['tests']),
        ('card_sol_original_p_note', r'(nominal $p=0.027$) over those 26 tests', orig['u']),
        ('card_sol_classified', r'Replay correction: 2,986 classified replays', int(interval['b1'] + interval['b0'])),
        ('card_placebo_note', r'200 full-size placebo runs (state labels permuted across all 410 stores, same prompt and agent) state 12 to 1,252 tests',
         [len(full), min(full), max(full)]),
        ('card_placebo_tests_quartiles', r'quartiles 24 and 36, median 30', [quartiles[0], statistics.median(full), quartiles[2]]),
    ]
    (DERIVED / 'numbers_card.json').write_text(json.dumps(numbers, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
