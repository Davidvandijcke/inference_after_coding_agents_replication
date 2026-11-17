#!/usr/bin/env python3
"""Table 9: how four agents' reports treat a significant lead (u <= 0.05) on the 100 open-ended null datasets.

Columns: the opening's stance (blind read), the median number of tests the reports state, the share of significant
leads the report corrects, and the share whose opening gives a corrected p-value (text search).

Inputs:  data/agent_runs/multi_model_originals/, data/agent_runs/open_ended_luna/originals.csv and reports.jsonl.gz
Outputs: paper/tabs/self_correction_stance.tex, data/derived/10_self_correction/numbers_stance.json
"""
import csv
import gzip
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'data' / 'agent_runs'
DERIVED = ROOT / 'data' / 'derived' / '10_self_correction'
MODELS = ('gpt-6-luna', 'gpt-6.1-sol', 'claude-sonnet-5-5', 'claude-opus-5-5')
NAME = {'gpt-6-luna': 'GPT-6-Luna', 'gpt-6.1-sol': 'GPT-6.1-Sol', 'claude-sonnet-5-5': 'Claude Sonnet 5.5', 'claude-opus-5-5': 'Claude Opus 5.5'}
OPENING_LIMIT = 800
NUM = r'(0?\.\d+|\d\.\d+e-?\d+|1\.0+|[<≈~]\s*0?\.\d+)'
ADJUSTED_P = re.compile(r'(holm|bonferroni|adjusted|corrected|permutation|fdr|family-?wise|multiplicity)[^.;]{0,80}?\bp(-?value)?\b[^.;0-9]{0,15}' + NUM
                        + r'|\bp(-?value)?\b[^.;0-9]{0,15}' + NUM + r'[^.;]{0,60}?(holm|bonferroni|adjusted|corrected|after (adjust|correct)|multiplicity)', re.I)


def opening(report):
    lines = [ln for ln in (report or '').splitlines() if not re.match(r'\s*#', ln)]
    text = ' '.join(' '.join(lines).replace('|', ' ').replace('*', '').replace('`', '').split())
    if len(text) <= OPENING_LIMIT:
        return text
    cut = text[:OPENING_LIMIT]
    end = max(cut.rfind('. '), cut.rfind('.) '), cut.rfind('? '))
    return cut[:end + 1] if end >= 200 else cut


def sessions():
    """(model, slot) -> u for usable originals."""
    out = {}
    for r in csv.DictReader(open(RUNS / 'multi_model_originals' / 'sessions.csv', encoding='utf-8')):
        if r['status'] == 'completed' and r['commit_model'] and r['u']:
            out[(r['model'], int(r['slot']))] = float(r['u'])
    for r in csv.DictReader(open(RUNS / 'open_ended_luna' / 'originals.csv', encoding='utf-8')):
        if r['usable'] == 'TRUE':
            out[('gpt-6-luna', int(r['slot']))] = float(r['u'])
    return out


def summarize():
    u = sessions()
    reads = {(r['model'], int(r['slot']), r['task']): r for r in csv.DictReader(open(RUNS / 'multi_model_originals' / 'blind_reads.csv', encoding='utf-8')) if r['read'] == '1'}
    with gzip.open(RUNS / 'multi_model_originals' / 'reports.jsonl.gz', 'rt', encoding='utf-8') as f:
        reports = {(r['model'], r['slot']): r['final_report'] for r in map(json.loads, f)}
    with gzip.open(RUNS / 'open_ended_luna' / 'reports.jsonl.gz', 'rt', encoding='utf-8') as f:
        reports.update({('gpt-6-luna', r['slot']): r['final_report'] for r in map(json.loads, f)})
    out = {}
    for m in MODELS:
        usable = sorted(s for (mm, s) in u if mm == m)
        sig = [s for s in usable if u[(m, s)] <= 0.05]
        facts = [reads[(m, s, 'facts')] for s in sig]
        corrected = [r for r in facts if r['own_correction'] != 'NONE']
        texts = [opening(reports[(m, s)]) for s in sig]
        out[m] = dict(usable=len(usable), significant=len(sig),
                      opening=Counter(reads[(m, s, 'opening')]['category'] for s in sig),
                      datasets=len({s for (mm, s, t) in reads if mm == m}),
                      tests_median=statistics.median(int(reads[(m, s, 'facts')]['tests_stated']) for s in usable if reads[(m, s, 'facts')]['tests_stated']),
                      corrected=len(corrected), adjusted_in_opening=sum(bool(ADJUSTED_P.search(t)) for t in texts),
                      # leads whose program returns the report's adjusted p-value instead of the nominal one
                      returned_adjusted=sum(abs(u[(m, s)] - float(r['adjusted_p'])) <= 1e-3 * float(r['adjusted_p'])
                                            for s, r in zip(sig, facts) if r['own_correction'] != 'NONE' and r['adjusted_p']))
    return out


def pct(k, n):
    return f'{100 * k / n:.0f}'


def table(p):
    rows = []
    for m in MODELS:
        s, n = p[m], p[m]['significant']
        rows.append(' & '.join([NAME[m], str(n), pct(s['opening']['STANDS_BEHIND'], n), pct(s['opening']['WARNS'], n), pct(s['opening']['DISCLAIMS'], n),
                                f"{s['tests_median']:.0f}", pct(s['corrected'], n), pct(s['adjusted_in_opening'], n)]) + r' \\')
    return (r'''\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{Disclaimer in the opening (\%)} & \multicolumn{3}{c}{Own multiplicity correction} \\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
Agent & \begin{tabular}[b]{@{}c@{}}Leads with\\$p\le0.05$\end{tabular} & None & Warning & Disclaimer & \begin{tabular}[b]{@{}c@{}}Tests stated\\(median)\end{tabular} & \begin{tabular}[b]{@{}c@{}}Applied\\(\% of leads)\end{tabular} & \begin{tabular}[b]{@{}c@{}}Given in the\\opening (\% of leads)\end{tabular} \\
\midrule
''' + '\n'.join(rows) + '\n' + r'''\bottomrule
\end{tabular}
''')


def numbers(p):
    luna, sol, son, opus = (p[m] for m in MODELS)
    sig = {m: 100 * p[m]['significant'] / p[m]['usable'] for m in MODELS}
    share = lambda k, s: 100 * s[k] / s['significant']
    none = {m: 100 * p[m]['opening']['STANDS_BEHIND'] / p[m]['significant'] for m in MODELS}
    in_opening = {m: share('adjusted_in_opening', p[m]) for m in MODELS}
    medians = [p[m]['tests_median'] for m in MODELS[1:]]
    return [
        ('null_significant_range_abstract', r'four frontier agents report significant $p$-values between 54\% and 98\% of the time',
         [min(sig.values()), max(sig.values())]),
        ('null_significant_opus_sol_intro', r'Claude Opus 5.5 and GPT-6.1-Sol (at high effort), report a significant ($p \leq 0.05$) result 54 and 88\% of the time',
         [sig['claude-opus-5-5'], sig['gpt-6.1-sol']]),
        ('null_significant_sonnet_opus', r'report significant $p$-values when all effects are zero 74 and 54\% of the time',
         [sig['claude-sonnet-5-5'], sig['claude-opus-5-5']]),
        ('null_datasets', r'on the 100 simulations of the open-ended design', min(p[m]['datasets'] for m in MODELS)),
        ('correct_nearly_all', r'Sol, Sonnet and Opus correct nearly all of their significant leads',
         [share('corrected', s) for s in (sol, son, opus)]),
        ('tests_stated_median_range', r'for a median of 312 to 378 tests counted by their own reports', [min(medians), max(medians)]),
        ('luna_corrects_share', r'while Luna corrects only 58\% of them', share('corrected', luna)),
        ('returned_p_never_corrected', r'is \textit{never} the corrected $p$-value', sum(p[m]['returned_adjusted'] for m in MODELS)),
        ('false_rejection_range', r'with 54--98\% of them falsely rejecting across the four agents', [min(sig.values()), max(sig.values())]),
        ('corrected_p_in_opening_range', r'the opening of the report gives the corrected $p$-value for only 7 to 25\% of significant leads',
         [min(in_opening.values()), max(in_opening.values())]),
        ('no_reason_to_doubt_range', r"the openings of 31--85\% of the different agents' significant reports give no concrete reason to doubt the finding",
         [min(none.values()), max(none.values())]),
        ('usable_reports', r'usable reports (no errors or refusals) 99, 100, 100 and 97', [p[m]['usable'] for m in MODELS]),
    ]

if __name__ == '__main__':
    p = summarize()
    (ROOT / 'paper' / 'tabs').mkdir(parents=True, exist_ok=True)
    (ROOT / 'paper' / 'tabs' / 'self_correction_stance.tex').write_text(table(p), encoding='utf-8')
    DERIVED.mkdir(parents=True, exist_ok=True)
    (DERIVED / 'numbers_stance.json').write_text(json.dumps(numbers(p), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
