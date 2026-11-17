#!/usr/bin/env python3
"""Figure 7 and its numbers: the purpose of each replay's steps over ten equal parts of its sequence. A step is a
shell command, with the purpose the annotators coded, or a file edit, classified by the files it changes. Each
replay has weight one, spread over its steps; replays with a command coded unknown are left out.

Inputs:  data/agent_runs/card_luna/activity-steps.csv.gz, replays.csv.gz
Outputs: paper/figs/card_activity_timeline.pdf, data/derived/08_card/activity.json
"""
import csv
import gzip
import json
from itertools import groupby
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'data' / 'agent_runs' / 'card_luna'
STEPS = ('Set up environment', 'Inspect data', 'Write analysis script', 'Estimate or evaluate',
         'Interpret or choose', 'Verify', 'Report', 'Debug or other')
COMMAND_STEP = {code: STEPS.index(step) for step, codes in (
    ('Set up environment', ('setup',)), ('Inspect data', ('inspect',)), ('Write analysis script', ('construct',)),
    ('Estimate or evaluate', ('evaluate',)), ('Interpret or choose', ('interpret', 'choose', 'revise')),
    ('Verify', ('verify',)), ('Report', ('report',)), ('Debug or other', ('debug', 'mixed', 'other')))
    for code in codes}
CODE_FILES = ('.r', '.py', '.ipynb', '.sh', '.rmd')
REPORT_FILES = ('.md', '.txt', '.tex', '.html')
BINS = 10


def file_step(paths):
    suffixes = {Path(p).suffix.lower() for p in paths}
    if suffixes & set(CODE_FILES): return STEPS.index('Write analysis script')
    if suffixes & set(REPORT_FILES): return STEPS.index('Report')
    return STEPS.index('Debug or other')


def sequences():
    """Each usable replay's steps in order, as (purpose index, paths changed), with its stratum."""
    with gzip.open(RUNS / 'replays.csv.gz', 'rt', encoding='utf-8') as f:
        stratum = {r['session_id']: r['stratum'] for r in csv.DictReader(f)}
    with gzip.open(RUNS / 'activity-steps.csv.gz', 'rt', encoding='utf-8') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (r['session_id'], int(r['step'])))
    out = {}
    for sid, steps in groupby(rows, key=lambda r: r['session_id']):
        steps = list(steps)
        if any(r['kind'] == 'command' and r['purpose'] == 'unknown' for r in steps):
            continue
        out[sid] = (stratum[sid], [(COMMAND_STEP[r['purpose']], []) if r['kind'] == 'command' else
                                   (file_step(r['paths'].split(';')), r['paths'].split(';')) for r in steps])
    return out


def bin_of(k, n):
    return min(int((k + .5) / n * BINS), BINS - 1)


def composition(seqs):
    mass = np.zeros((len(STEPS), BINS))
    for seq in seqs:
        for k, (step, _) in enumerate(seq):
            mass[step, bin_of(k, len(seq))] += 1 / len(seq)
    return np.round(mass / mass.sum(axis=0), 10)


def first(seq, step):
    return next((k for k, (s, _) in enumerate(seq) if s == step), len(seq))


def timing(seqs):
    """Shares of replays: inspecting data in the first part, only setting up there, running an analysis before
    writing a script, and editing analysis.R after the first reporting step."""
    inspect, setup = STEPS.index('Inspect data'), STEPS.index('Set up environment')
    early = [[s for k, (s, _) in enumerate(seq) if bin_of(k, len(seq)) == 0] for seq in seqs]
    late_edit = lambda seq: any(k > first(seq, STEPS.index('Report')) and any(Path(p).name == 'analysis.R' for p in paths)
                                for k, (_, paths) in enumerate(seq))
    n = len(seqs)
    return {'inspect_in_first_part': sum(inspect in e for e in early) / n,
            'only_setup_in_first_part': sum(bool(e) and set(e) == {setup} for e in early) / n,
            'estimate_before_script': sum(first(seq, STEPS.index('Estimate or evaluate')) < first(seq, STEPS.index('Write analysis script'))
                                          for seq in seqs) / n,
            'analysis_R_edit_after_report': sum(late_edit(seq) for seq in seqs) / n}


def figure(comp, n, path):
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
                         'axes.linewidth': .5, 'savefig.facecolor': 'white'})
    colors = ['#4a5560', '#6da7ec', '#1baf7a', '#1c5cab', '#e34948', '#eda100', '#6250d6', '#b4b9be']
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    fig.subplots_adjust(left=.1, right=.69, top=.90, bottom=.17)
    bottom = np.zeros(BINS)
    for j, label in enumerate(STEPS):
        # no white outline on the thin red segments, so they keep their height
        edge = 'white' if label != 'Interpret or choose' else colors[j]
        ax.bar(np.arange(BINS) + .5, comp[j] * 100, bottom=bottom, width=1, color=colors[j], edgecolor=edge,
               linewidth=.2, label=label)
        bottom += comp[j] * 100
    ax.set_title(f'All replays (N = {n:,})', loc='left')
    ax.set_xlim(0, 10); ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100])
    ax.set_xticks([0, 5, 10], ['Start', 'Middle', 'End']); ax.set_xlabel("Position in the replay's sequence of steps")
    ax.set_ylabel('Share of steps (%)')
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], names[::-1], loc='center left', bbox_to_anchor=(.71, .53), ncol=1, frameon=False,
               fontsize=6.8, handlelength=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)


def main():
    usable = sequences()
    seqs = [seq for _, seq in usable.values()]
    comp = composition(seqs)
    figure(comp, len(seqs), ROOT / 'paper/figs/card_activity_timeline.pdf')
    summary = {'replays': {s: sum(st == s for st, _ in usable.values()) for s in ('tail', 'body')},
               'steps': list(STEPS), 'composition': comp.tolist(), 'timing': timing(seqs)}
    (ROOT / 'data/derived/08_card/activity.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
