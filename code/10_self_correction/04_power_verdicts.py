#!/usr/bin/env python3
"""Figure 10: GPT-6.1-Sol's own correction against the conditional correction on planted datasets with a real effect.

Own correction as written: an uncorrected lead is kept when p <= 0.05, a corrected lead when the report says it
survives. Conditional correction: (1 + #{null p <= u}) / (K + 1) over the K fresh null runs whose program is the
common analysis and whose declaration the blind reader matched to it.

Inputs:  data/agent_runs/sol_common_analysis/planted_sessions.csv, reference_sessions.csv
Outputs: paper/figs/self_correction_verdicts.pdf, data/derived/10_self_correction/power_cases.csv, numbers_power.json
"""
import csv
import json
import os
from math import comb
from pathlib import Path

os.environ['SOURCE_DATE_EPOCH'] = '0'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / 'data' / 'agent_runs' / 'sol_common_analysis'
DERIVED = ROOT / 'data' / 'derived' / '10_self_correction'
WINE = '#882255'
FAMILY = 18  # the family Sol's corrections state


def common_procedure(r):
    """Employment at 12 months, all participants, treatment and site indicators only, HC2 errors, Student-t test."""
    return (r['outcome'] == 'employed' and r['followup'] == '12' and r['restricted'] == 'False' and r['interaction'] == 'False'
            and r['baseline_columns'] == '0' and r['robust_se'] == 'HC2' and r['reference_distribution'] == 'student_t')


def exact_conditional_mean(declared_model):
    """A declaration of an exact conditional-mean model selects a different analysis."""
    return 'E[Y' in declared_model and 'does not impose' not in declared_model


def cases():
    refs = [r for r in csv.DictReader(open(COHORT / 'reference_sessions.csv', encoding='utf-8'))]
    bank = sorted(float(r['u']) for r in refs if r['common_procedure'] == 'True' and r['reader_label'] == 'MATCH')
    rows = []
    for r in csv.DictReader(open(COHORT / 'planted_sessions.csv', encoding='utf-8')):
        if not common_procedure(r) or exact_conditional_mean(r['commit_model']):
            continue
        u = float(r['u'])
        own = u <= 0.05 if r['own_correction'] == 'NONE' else r['survives'] == 'YES'
        p_cond = (1 + sum(p <= u for p in bank)) / (len(bank) + 1)
        rows.append(dict(cohort=r['cohort'], slot=int(r['slot']), u=u, own_correction=r['own_correction'], correction_tests=r['correction_tests'],
                         tests_stated=r['tests_stated'], adjusted_p=r['adjusted_p'], survives=r['survives'], own_keep=own,
                         p_conditional=p_cond, conditional_keep=p_cond <= 0.05))
    return sorted(rows, key=lambda r: (r['cohort'], r['slot'])), bank, len(refs)


def mcnemar_exact(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, j) for j in range(min(a, b) + 1)) / 2 ** n)


def verdict(r):
    assert not (r['own_keep'] and not r['conditional_keep']), 'no dataset is kept by the own correction alone'
    return 'both' if r['own_keep'] else 'conditional' if r['conditional_keep'] else 'neither'


def figure(rows, path):
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#777777', 'axes.linewidth': .6, 'pdf.fonttype': 42})
    level = {'both': 2, 'conditional': 1, 'neither': 0}
    fig, b = plt.subplots(figsize=(5.4, 2.6))
    for corrected, mfc, label in ((True, WINE, 'run applied its own correction'), (False, 'white', 'run applied no correction')):
        sel = [r for r in rows if (r['own_correction'] != 'NONE') == corrected]
        b.plot([r['u'] for r in sel], [level[verdict(r)] for r in sel], 'o', color=WINE, mfc=mfc, mec=WINE, mew=1.1, ms=5.5, zorder=3, label=label)
    counts = {k: sum(verdict(r) == k for r in rows) for k in level}
    b.set_xscale('log')
    b.set_xlim(5e-7, 0.6)
    b.set_ylim(-0.7, 2.7)
    b.set_yticks([2, 1, 0])
    b.set_yticklabels([f"kept by both ({counts['both']})", f"kept by the conditional\ncorrection only ({counts['conditional']})", f"kept by neither ({counts['neither']})"])
    for v, txt in ((0.05 / FAMILY, f'0.05/{FAMILY}'), (0.05, '0.05')):
        b.axvline(v, ls=':', lw=1, color='.45', zorder=1)
        b.text(v, 2.62, txt, fontsize=7.5, ha='center', va='bottom', color='.3')
    b.set_xlabel('Nominal $p$-value of the lead')
    b.grid(axis='x', color='.92', lw=.7, zorder=0)
    b.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, -.5), ncol=2, handletextpad=.4, columnspacing=1.5)
    fig.subplots_adjust(bottom=.33, left=.3, right=.98, top=.9)
    fig.savefig(path, metadata={'Title': "GPT-6.1-Sol's own correction against the conditional correction on 49 real effects", 'CreationDate': None})
    plt.close(fig)


def main():
    rows, bank, n_refs = cases()
    corrected = [r for r in rows if r['own_correction'] != 'NONE']
    uncorrected = [r for r in rows if r['own_correction'] == 'NONE']
    gains = [r for r in rows if r['conditional_keep'] and not r['own_keep']]
    losses = [r for r in rows if r['own_keep'] and not r['conditional_keep']]
    lo, hi = min(r['u'] for r in gains), max(r['u'] for r in gains)
    assert not losses and lo > 0.05 / FAMILY
    holm_bonferroni = sum(r['own_correction'] in ('HOLM', 'BONFERRONI') and r['correction_tests'] == str(FAMILY) for r in corrected)
    DERIVED.mkdir(parents=True, exist_ok=True)
    with open(DERIVED / 'power_cases.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows({**r, 'u': repr(r['u']), 'p_conditional': repr(r['p_conditional'])} for r in rows)
    figure(rows, ROOT / 'paper' / 'figs' / 'self_correction_verdicts.pdf')
    numbers = [
        ('power_datasets', r'on 49 simulated datasets with a moderate true effect', len(rows)),
        ('power_corrected_runs', r'40 runs corrected over 18 tests', sum(r['tests_stated'] == str(FAMILY) for r in corrected)),
        ('power_uncorrected_runs', r'and 9 applied no correction', len(uncorrected)),
        ('power_own_keeps', r'The corrections as written keep 30 of the 49 effects', sum(r['own_keep'] for r in rows)),
        ('power_conditional_keeps', r'against 39 for the correction conditional on reporting that analysis', sum(r['conditional_keep'] for r in rows)),
        ('power_gains_losses', r'(nine gains and no losses', [len(gains), len(losses)]),
        ('power_mcnemar', r'exact paired $p=0.004$', mcnemar_exact(len(gains), len(losses))),
        ('power_gain_p_range', r"All nine gains have nominal $p$-values between 0.003 and 0.02, above Holm's threshold of $0.05/18$", [lo, hi]),
        ('power_uncorrected_same_range', r'three runs that did not correct kept findings with $p$-values in the same range',
         sum(lo <= r['u'] <= hi and r['own_keep'] for r in uncorrected)),
        ('power_datasets_note', r'49 datasets of the open-ended simulation design with a moderate true effect', len(rows)),
        ('power_caption_corrected', r'Filled: agent estimated a Holm or Bonferroni correction over a stated family of 18 (40 runs)', holm_bonferroni),
        ('power_caption_uncorrected', r'hollow: no correction, nominal $p$ as written (9 runs)', len(uncorrected)),
        ('power_reference_bank', r'among the 51 of 100 fresh null runs that reported the same analysis', [len(bank), n_refs]),
    ]
    (DERIVED / 'numbers_power.json').write_text(json.dumps(numbers, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
