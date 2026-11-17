"""Table 1 Panel B, Figure 3 and the open-ended design's in-text numbers.

Panel B: rejection rates of the agent's p-value and of the replay test, re-selection rates, interval width and
multiplier at 40 + 200 and 240 + 400 replays. Figure 3(a): rejection rates in the fixed-rule, open-ended and
real-data designs; 3(b): corrected against reported p-value at 240 + 400 replays.

Inputs:  data/derived/02_open_ended/, data/derived/01_fixed_design/fixed_rejections.csv,
         data/derived/03_placebo/per_original.csv, data/raw/pathways/
Outputs: paper/tabs/simulation_panel_b.tex, paper/figs/simulation_overview.pdf, results/numbers/02_open_ended.json
"""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'lib'))
from paper_numbers import write_numbers  # noqa: E402

DERIVED = ROOT / 'data' / 'derived'
BLUE, WINE, ORANGE = '#355f8d', '#8b245c', '#b36b24'
BUDGETS = ('base', 'pooled')  # 40 + 200 replays; 240 + 400 with the extension


def median(xs):
    xs = sorted(xs)
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2


def num(x):
    return None if x == '' else float(x)


def reject(upper, alpha=0.05):
    return upper < alpha


def pct(v):
    return f'{100 * v:.1f}\\%'


design = json.loads((DERIVED / '02_open_ended' / 'design.json').read_text(encoding='utf-8'))
N, M = design['n'], design['m']
counts = list(csv.DictReader(open(DERIVED / '02_open_ended' / 'replay_counts.csv', encoding='utf-8')))
# tail and body replays each original received
drawn = {c: {int(r['slot']): [int(r['b1']) + int(r['unclassified1']), int(r['b0']) + int(r['unclassified0'])]
             for r in counts if r['component'] == c} for c in ('base', 'ext1')}
drawn['pooled'] = {s: [a + b for a, b in zip(v, drawn['ext1'][s])] for s, v in drawn['base'].items()}
nominal = {c: [max(v[i] for v in d.values()) for i in (0, 1)] for c, d in drawn.items()}  # as designed
fewest = {c: [min(v[i] for v in d.values()) for i in (0, 1)] for c, d in drawn.items()}   # a small bank gives fewer
LABELS = {b: f'{nominal[b][0]} + {nominal[b][1]}' for b in BUDGETS}

rows = {b: [] for b in BUDGETS}
for r in csv.DictReader(open(DERIVED / '02_open_ended' / 'per_original.csv', encoding='utf-8')):
    rows[r['budget']].append({k: (r[k] if k in ('budget', 'status') else int(r[k]) if k in ('slot', 'attempted', 'x1', 'b1', 'x0', 'b0')
                                  else num(r[k])) for k in r})


def budget_stats(z):
    with_tail = [r for r in z if r['x1'] > 0]
    both = [r for r in with_tail if r['x0'] > 0]
    return dict(
        n=len(z), attempted=sum(r['attempted'] for r in z),
        agent=sum(r['u'] <= .05 for r in z), replay=sum(reject(r['upper_one95']) for r in z),
        x1=sum(r['x1'] for r in z), b1=sum(r['b1'] for r in z), x0=sum(r['x0'] for r in z), b0=sum(r['b0'] for r in z),
        neither=sum(r['x1'] == 0 and r['x0'] == 0 for r in z),
        mean_width=sum(r['upper95'] - r['lower95'] for r in with_tail) / len(with_tail),
        mean_width_both=sum(r['upper95'] - r['lower95'] for r in both) / len(both),
        median_lambda=median([r['multiplier'] for r in both]),
        lambda_lower=median([r['lower95'] / r['u'] for r in both]),
        lambda_upper=median([r['upper95'] / r['u'] for r in both]))


stats = {b: budget_stats(rows[b]) for b in BUDGETS}

# Table 1, Panel B
N_TEX = f'{N:,}'.replace(',', '{,}')
tex = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccccc@{}}',
       f"\\multicolumn{{8}}{{@{{}}l}}{{\\rlap{{\\textit{{Panel B. Open-ended agents, simulated data: {stats['base']['n']} "
       f"datasets, $n = {N_TEX}$, $m = {M}$, replays tail + body}}}}}} \\\\",
       r'\addlinespace[2pt]',
       r' & \multicolumn{2}{c}{Reject at 5\%} & \multicolumn{2}{c}{Re-selected} & & \multicolumn{2}{c}{Multiplier $\widehat\lambda$} \\',
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){7-8}',
       r"Replays & Agent's $p$ & Replay test & Tail & Body & Width & Median & 95\% interval \\", r'\midrule']
for b in BUDGETS:
    q = stats[b]
    tex.append(f"{LABELS[b]} & {pct(q['agent'] / q['n'])} & {pct(q['replay'] / q['n'])} & {pct(q['x1'] / q['b1'])} & "
               f"{100 * q['x0'] / q['b0']:.2f}\\% & {q['mean_width']:.2f} & {q['median_lambda']:.0f} & "
               f"[{q['lambda_lower']:.1f}, {q['lambda_upper']:.0f}] \\\\")
tex.append(r'\end{tabular*}')
(ROOT / 'paper' / 'tabs').mkdir(parents=True, exist_ok=True)
(ROOT / 'paper' / 'tabs' / 'simulation_panel_b.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')

# Figure 3
fixed = [r for r in csv.DictReader(open(DERIVED / '01_fixed_design' / 'fixed_rejections.csv', encoding='utf-8'))
         if r['in_figure_3a'] == 'TRUE']
fx = {k: sum(r[k] == 'TRUE' for r in fixed) / len(fixed) for k in ('reject_agent_05', 'reject_replay_05', 'reject_oracle_05')}
placebo = list(csv.DictReader(open(DERIVED / '03_placebo' / 'per_original.csv', encoding='utf-8')))
pl = {k: sum(int(r[k]) for r in placebo) / len(placebo) for k in ('reject_agent_05', 'reject_replay_05')}
e = stats['pooled']

os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'axes.titlesize': 12})
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={'width_ratios': [1, 1.1]})

a = ax[0]
labels = [f"Fixed-rule selector\n{len(fixed)} datasets\noracle known",
          f"Open-ended agents\n{e['n']} analyses\n{LABELS['pooled']} replays",
          f"Real data, {len({r['study'] for r in placebo})} studies\n{len(placebo)} analyses\n"
          f"{max(int(r['b1']) for r in placebo)} + {max(int(r['b0']) for r in placebo)} replays"]
nom = [fx['reject_agent_05'], e['agent'] / e['n'], pl['reject_agent_05']]
ub = [fx['reject_replay_05'], e['replay'] / e['n'], pl['reject_replay_05']]
y = [2.0, 1.0, 0.0]
h = 0.32
a.barh([v + h / 2 + .02 for v in y], nom, h, color=WINE, label="Agent's own $p$-value")
a.barh([v - h / 2 - .02 for v in y], ub, h, color=BLUE, label='Replay test')
for yy, v in zip([v + h / 2 + .02 for v in y], nom):
    a.text(v + .012, yy, f'{v:.1%}', va='center', color=WINE, fontsize=9)
for yy, v in zip([v - h / 2 - .02 for v in y], ub):
    a.text(max(v, .065) + .012, yy, f'{v:.1%}', va='center', color=BLUE, fontsize=9)
a.axvline(.05, ls=':', lw=1, color='.4')
oracle = fx['reject_oracle_05']
a.axvline(oracle, ymin=.7, ymax=1, ls='--', lw=1, color='.55')
a.text(oracle + .012, y[0] + 2 * h + .12, f'Fixed-rule oracle: {oracle:.2%}', fontsize=8.5, color='.35', va='bottom')
a.set(yticks=y, yticklabels=labels, xlim=(0, 1.0), ylim=(-.5, 2.85))
a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
a.set_xlabel('Rejection of a true null at the 5% level')
a.set_title('(a)', loc='left')
a.legend(loc='upper left', bbox_to_anchor=(0, -.2), frameon=False, ncol=1, fontsize=9)
a.grid(axis='x', color='.9', lw=.8)
a.set_axisbelow(True)

# Panel (b): rejected analyses are diamonds, hollow and at the bound when there is no point estimate.
b = ax[1]
inf, nob, una, rej, bound_rej = [], [], [], [], []
for r in rows['pooled']:
    if reject(r['upper_one95']) and r['x1'] > 0 and r['x0'] > 0:
        rej.append((r['u'], r['p_corrected']))
    elif reject(r['upper_one95']):
        bound_rej.append((r['u'], r['upper_one95']))
    elif r['x1'] == 0:
        if r['x0'] > 0:
            sys.exit(f"Figure 3(b) has no symbol for an analysis with a body match only that the replay test does "
                     f"not reject (slot {r['slot']})")
        una.append((r['u'], 1.0))
    elif r['x0'] == 0:
        nob.append((r['u'], 1.0))
    else:
        inf.append((r['u'], r['p_corrected']))
empty = [name for name, points in (('no body match', nob), ('no tail or body match', una), ('not rejected', inf),
                                   ('rejected', rej), ('rejected, no tail match', bound_rej)) if not points]
if empty:
    sys.exit('Figure 3(b) and its legend expect analyses in every category; none in: ' + ', '.join(empty))
lam = e['median_lambda']
xs = [1e-4, 1]
b.plot(xs, xs, ls='--', lw=1, color='.5', zorder=1)
b.plot([x for x in xs if x * lam <= 1] + [1 / lam], [x * lam for x in xs if x * lam <= 1] + [1], ls='-.', lw=1,
       color=ORANGE, zorder=1, label=f"Median multiplier {lam:.0f}")
b.axvline(.05, ls=':', lw=1, color='.4', zorder=1)
b.axhline(.05, ls=':', lw=1, color='.4', zorder=1)
kw = dict(zorder=3, clip_on=False)
b.scatter(*zip(*nob), s=28, color='.6', label=f'No body match, $p=1$ ({len(nob)})', **kw)
b.scatter(*zip(*una), s=28, facecolor='white', edgecolor='.4', lw=.9, label=f'No tail or body match, NA ({len(una)})', **kw)
b.scatter(*zip(*inf), s=30, color=BLUE, label=f'Not rejected ({len(inf)})', **kw)
b.scatter(*zip(*rej), s=42, color=WINE, marker='D', label=f'Rejected ({len(rej)})', **kw)
b.scatter(*zip(*bound_rej), s=42, facecolor='white', edgecolor=WINE, lw=1.1, marker='D',
          label=f'Rejected, no tail match ({len(bound_rej)})', **kw)
b.text(1.15e-4, .6, f'{len(una) + len(nob)} of {e["n"]} at $p=1$', fontsize=8, color='.4')
b.set(xscale='log', yscale='log', xlim=(1e-4, 1), ylim=(1e-3, 1))
b.set_xlabel("Agent's own $p$-value")
b.set_ylabel('Corrected $p$-value')
b.set_title('(b)', loc='left')
b.text(2.6e-3, 2.6e-3 * 1.25, 'no correction', fontsize=8, color='.45', rotation=45, rotation_mode='anchor', va='bottom')
b.text(.052, 1.2e-3, '5%', fontsize=8, color='.4')
b.text(1.1e-4, .056, '5%', fontsize=8, color='.4')
b.legend(loc='upper left', bbox_to_anchor=(-.24, -.2), frameon=False, ncol=2, fontsize=8, handlelength=1.6, columnspacing=.6)
fig.subplots_adjust(left=.19, right=.98, bottom=.29, top=.91, wspace=.4)
(ROOT / 'paper' / 'figs').mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / 'paper' / 'figs' / 'simulation_overview.pdf', metadata={'CreationDate': None, 'ModDate': None})
plt.close(fig)

# In-text numbers
p = stats['base']
datasets = sorted((ROOT / 'data' / 'raw' / 'pathways').glob('main-*.csv'))
sites, floor_half = set(), True
for f in datasets:
    data = list(csv.DictReader(open(f, encoding='utf-8')))
    for site in {r['site'] for r in data}:
        k = [r['treatment'] for r in data if r['site'] == site]
        sites.add(site)
        floor_half &= k.count('1') == len(k) // 2

write_numbers('02_open_ended', [
    ('oe_datasets', 'we simulate 100 datasets from an imaginary randomized control trial', len(datasets)),
    ('oe_participants', 'The program has $2{,}000$ participants', N),
    ('oe_sites', 'participants at four sites', len(sites)),
    ('oe_half_treated', "with half of each site's participants assigned to the program at random", floor_half),
    ('oe_m', 'of size $m=200$, giving $m/n=0.1$', [M, M / N]),
    ('oe_budgets', 'We draw $40+200$ and $240+400$ tail and body subsamples respectively',
     [fewest['base'], fewest['pooled']]),
    ('oe_agent_reject', 'reporting a $p$-value of less than $0.05$ for nearly every original analysis',
     e['agent'] / e['n']),
    ('oe_replay_reject', r'5.1\% for the open-ended agents', e['replay'] / e['n']),
    ('oe_median_lambda', r'a multiplier $\widehat \lambda$ of 19, our median estimated multiplier', e['median_lambda']),
    ('oe_hollow_diamond', 'as in the hollow red diamond on the plot', len(bound_rej)),
    ('oe_neither_match', 'In 19 out of 99 analyses, there is neither a tail nor a body match', [e['neither'], e['n']]),
    ('oe_figure_design', '2) the open-ended design (99 datasets ... $m/n=0.10$) with 240 tail and 400 body replays',
     [e['n'], M / N] + nominal['pooled']),
    ('oe_figure_median_lambda', 'the dash-dotted line marks the median multiplier of 19 over the analyses '
     're-selected in both strata', e['median_lambda']),
    ('oe_usable', 'so we dropped it and report the results from the remaining 99', e['n']),
    ('oe_agent_reject_count', r'rejecting in 97/99 (98.0\% rounded)', [e['agent'], e['n']]),
    ('oe_replay_reject_row2', r'Panel B, row 2, shows ... with enough replays (5.1\%)', e['replay'] / e['n']),
    ('oe_width', 'The fairly large width of $0.71$', e['mean_width']),
    ('oe_width_both', 'the mean width drops to $0.44$ at 640 replays', [e['mean_width_both'], sum(nominal['pooled'])]),
    ('oe_body_reselect', r'(only 0.29\% of body replays)', p['x0'] / p['b0']),
    ('oe_median_lambda_range', 'with median multipliers of $17$--$19$', [p['median_lambda'], e['median_lambda']]),
    ('oe_lambda_lower_range', '(lower confidence bounds $6.9$--$7.2$)', [p['lambda_lower'], e['lambda_lower']]),
    ('oe_appendix_n', r'Each dataset has $n = 2{,}000$ participants', N),
    ('oe_appendix_half', "Treatment $T$ equals one for exactly half of each site's participants", floor_half),
])
