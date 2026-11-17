#!/usr/bin/env python3
"""Figure 6: how often the replays chose each employment measure and specification when that option was, or was
not, the most significant one on the replay's subset (ties go to the option listed first). Each bar pair splits all
2,989 replays, pooled as run; a replay whose coded choice is neither option counts as not choosing it.

Input:   data/derived/08_card/choice-pvalues.csv
Outputs: paper/figs/card_choice_significance.pdf, data/derived/08_card/choice-summary.json
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / 'data' / 'derived' / '08_card'
BLUE, PALE, INK, MUTED, GRID = '#335e8c', '#b9c6d4', '#29333d', '#5f6872', '#e6e8eb'
GROUPS = [
    ('Employment measure', 'measure',
     [('nonmanager_fte', 'Nonmanager FTE (original)'), ('full_time_employees', 'Full-time employees'),
      ('employee_headcount', 'Total headcount'), ('manager_inclusive_fte', 'FTE including managers')]),
    ('Specification', 'specification',
     [('nj_additive_chain', 'NJ and chain indicators (original)'), ('nj_only', 'NJ indicator only')]),
]


def summarize(rows):
    order = [o for o, _ in GROUPS[0][2]]
    for r in rows:
        menu = {k: float(r['p_' + k]) for k in order}
        r['best_measure'] = min(menu, key=lambda k: (menu[k], order.index(k)))
        r['best_specification'] = 'nj_only' if float(r['p_nj_only']) < float(r['p_nj_additive_chain']) else 'nj_additive_chain'
    out = {'replays': len(rows)}
    for _, key, options in GROUPS:
        out[key] = []
        for k, label in options:
            yes = [r for r in rows if r['best_' + key] == k]; no = [r for r in rows if r['best_' + key] != k]
            out[key].append({'option': k, 'label': label,
                             'chosen_when_most_significant': sum(r[key] == k for r in yes) / len(yes),
                             'n_most_significant': len(yes),
                             'chosen_otherwise': sum(r[key] == k for r in no) / len(no), 'n_otherwise': len(no)})
    return out


def render(summary, path):
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.edgecolor': MUTED})
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    h, y, ticks, labels = 0.34, 0.0, [], []
    for g, (title, key, _) in enumerate(GROUPS):
        if g:
            y -= 0.75
        ax.text(-0.01, y + 0.15, title, transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=8.4, weight='bold', color=INK)
        y -= 0.55
        for o in summary[key]:
            a, b = 100 * o['chosen_when_most_significant'], 100 * o['chosen_otherwise']
            ax.barh(y + h / 2, a, height=h, color=BLUE)
            ax.barh(y - h / 2, b, height=h, color=PALE)
            ax.text(a + 1, y + h / 2, f'{a:.0f}%', va='center', fontsize=7.6, color=BLUE)
            ax.text(b + 1, y - h / 2, f'{b:.0f}%', va='center', fontsize=7.6, color=MUTED)
            ticks.append(y); labels.append(o['label']); y -= 1
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.9, color=INK)
    ax.set_ylim(y + 0.45, 0.55); ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels([f'{t}%' for t in range(0, 101, 25)], color=MUTED)
    ax.set_xlabel('Share of replays that chose the option', color=MUTED, fontsize=7.9)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.legend(handles=[Patch(fc=BLUE, label="The option was the most significant on the replay's data"),
                        Patch(fc=PALE, label='Another option was more significant')],
               loc='upper left', bbox_to_anchor=(0.385, 0.995), frameon=False, fontsize=7.7)
    fig.subplots_adjust(left=0.385, right=0.97, top=0.86, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)


def main():
    rows = list(csv.DictReader(open(DERIVED / 'choice-pvalues.csv', encoding='utf-8')))
    summary = summarize(rows)
    render(summary, ROOT / 'paper/figs/card_choice_significance.pdf')
    (DERIVED / 'choice-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
