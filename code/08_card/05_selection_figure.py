#!/usr/bin/env python3
"""Figure 5: (a) the selection multiplier with its one-sided 95% lower bound; (b) the original p-value and the
corrected one with its one-sided 95% upper bound.

Input:   data/derived/08_card/correction.csv
Output:  paper/figs/card_selection.pdf
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
BLUE, GREY = '#335e8c', '#777777'


def main():
    c = {r['name']: float(r['value']) for r in csv.DictReader(open(ROOT / 'data/derived/08_card/correction.csv', encoding='utf-8'))}
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 9,
                         'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': GREY, 'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 1.9), layout='constrained')
    for a in ax:
        a.spines['left'].set_visible(False); a.set_ylim(-.25, .30); a.set_yticks([])
        a.grid(axis='x', color='#e3e3e3', lw=.5); a.set_axisbelow(True)
    lam, lo = c['multiplier'], c['multiplier_lower_one95']
    ax[0].errorbar(lam, 0, xerr=[[lam - lo], [0]], color=BLUE, fmt='o', markersize=4.5, lw=1.4, capsize=3, zorder=3)
    ax[0].annotate(f'{lam:.2f}', (lam, 0), xytext=(0, 9), textcoords='offset points', ha='center', va='bottom',
                   fontsize=8, color=BLUE)
    ax[0].axvline(1, color=GREY, lw=.8, ls='--'); ax[0].set_xlim(.8, 2.5)
    ax[0].set_xticks([1, 1.5, 2, 2.5], ['1', '1.5', '2', '2.5']); ax[0].set_xlabel(r'Selection multiplier $\lambda$')
    ax[0].set_title('(a) Selection multiplier', loc='left', pad=10)
    point, upper, u = c['p_corrected'], c['upper_one95'], c['u']
    ax[1].errorbar(point, .035, xerr=[[0], [upper - point]], color=BLUE, fmt='o', markersize=4.5, lw=1.4, capsize=3,
                   zorder=3)
    ax[1].scatter([u], [-.035], marker='D', s=28, facecolor='white', edgecolor=GREY, lw=1, zorder=4)
    for value, y, dy, color in [(point, .035, 9, BLUE), (u, -.035, -10, GREY)]:
        ax[1].annotate(f'{value:.4f}', (value, y), xytext=(0, dy), textcoords='offset points', ha='center',
                       va='bottom' if dy > 0 else 'top', fontsize=8, color=color)
    ax[1].axvline(.05, color=GREY, lw=.8, ls=':'); ax[1].set_xlim(0, .075)
    ax[1].set_xticks([0, .025, .05, .075], ['0', '0.025', '0.050', '0.075']); ax[1].set_xlabel(r'$p$-value')
    ax[1].set_title(r'(b) Nominal and adjusted $p$-values', loc='left', pad=10)
    fig.legend(handles=[Line2D([], [], marker='o', color=BLUE, ms=4.5, lw=1.4, label='Replay estimate (one-sided 95% bound)'),
                        Line2D([], [], marker='D', color=GREY, mfc='white', ms=4.5, lw=0, label=r'Nominal $p$-value')],
               frameon=False, loc='outside lower center', ncol=2, columnspacing=1.6, handlelength=1.6)
    path = ROOT / 'paper/figs/card_selection.pdf'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
