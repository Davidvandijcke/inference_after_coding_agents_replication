"""Table 1 Panel A and the fixed-rule design's in-text numbers.

Per budget of 25, 100, 400 and 1,000 replays per stratum: the replay test's rejection rate at 5%, its agreement
with the oracle, coverage of the exact selective p-value by the 95% interval, RMSE, mean width, and the median
multiplier with the medians of the interval ends divided by u.

Inputs:  data/derived/01_fixed_design/fixed_rejections.csv
Outputs: paper/tabs/simulation_panel_a.tex, results/numbers/01_fixed_design.json
"""
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'lib'))
from paper_numbers import write_numbers  # noqa: E402


def median(xs):
    xs = sorted(xs)
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2


def pct(v):
    return f'{100 * v:.1f}\\%'


rows = list(csv.DictReader(open(ROOT / 'data' / 'derived' / '01_fixed_design' / 'fixed_rejections.csv', encoding='utf-8')))
design = {k: {r[k] for r in rows} for k in ('n', 'm', 'k', 'epsilon')}
assert all(len(v) == 1 for v in design.values())
N, M, K = (int(design[k].pop()) for k in ('n', 'm', 'k'))
EPSILON = float(design['epsilon'].pop())
BUDGETS = sorted({int(r['b']) for r in rows})
N_DATASETS = len({r['case_id'] for r in rows})
(FIGURE_BUDGET,) = {int(r['b']) for r in rows if r['in_figure_3a'] == 'TRUE'}
stats = []
for b in BUDGETS:
    z = [r for r in rows if int(r['b']) == b]
    assert len(z) == N_DATASETS
    f = {k: [float(r[k]) for r in z] for k in ('u', 'oracle', 'p_corrected', 'lower95', 'upper95')}
    flag = {k: [r[k] == 'TRUE' for r in z] for k in ('reject_oracle_05', 'reject_replay_05')}
    n = len(z)
    stats.append(dict(
        b=b, n=n, replay=sum(flag['reject_replay_05']),
        agree=sum(a == o for a, o in zip(flag['reject_replay_05'], flag['reject_oracle_05'])),
        coverage=sum(lo <= o <= hi for lo, o, hi in zip(f['lower95'], f['oracle'], f['upper95'])),
        rmse=math.sqrt(sum((p - o) ** 2 for p, o in zip(f['p_corrected'], f['oracle'])) / n),
        mean_width=sum(hi - lo for lo, hi in zip(f['lower95'], f['upper95'])) / n,
        median_lambda=median([p / u for p, u in zip(f['p_corrected'], f['u'])]),
        lambda_lower=median([lo / u for lo, u in zip(f['lower95'], f['u'])]),
        lambda_upper=median([hi / u for hi, u in zip(f['upper95'], f['u'])])))

N_TEX = f'{N:,}'.replace(',', '{,}')
tex = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccccc@{}}', r'\toprule',
       f"\\multicolumn{{8}}{{@{{}}l}}{{\\rlap{{\\textit{{Panel A. Fixed-rule selector: {N_DATASETS} datasets, "
       f"$n = {N_TEX}$, $m = {M}$, replays per stratum}}}}}} \\\\",
       r'\addlinespace[2pt]',
       r' & \multicolumn{5}{c}{Replay test and corrected $p$-value} & \multicolumn{2}{c}{Multiplier $\widehat\lambda$} \\',
       r'\cmidrule(lr){2-6}\cmidrule(l){7-8}',
       r'Replays & Reject 5\% & Same as oracle & Coverage & RMSE & Width & Median & 95\% interval \\', r'\midrule']
for g in stats:
    tex.append(f"{g['b']:,} & {pct(g['replay'] / g['n'])} & {pct(g['agree'] / g['n'])} & {pct(g['coverage'] / g['n'])} & "
               f"{g['rmse']:.3f} & {g['mean_width']:.3f} & {g['median_lambda']:.1f} & "
               f"[{g['lambda_lower']:.1f}, {g['lambda_upper']:.1f}] \\\\")
tex.append(r'\end{tabular*}')
(ROOT / 'paper' / 'tabs').mkdir(parents=True, exist_ok=True)
(ROOT / 'paper' / 'tabs' / 'simulation_panel_a.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')

by_b = {g['b']: g for g in stats}
main, low, high = by_b[FIGURE_BUDGET], by_b[BUDGETS[0]], by_b[BUDGETS[-1]]
printed_lambda = [[g['median_lambda'], g['lambda_lower'], g['lambda_upper']] for g in stats]
coverage = [g['coverage'] / g['n'] for g in stats]
lambdas = [g['median_lambda'] for g in stats]
write_numbers('01_fixed_design', [
    ('fixed_datasets', 'we draw $400$ datasets', N_DATASETS),
    ('fixed_n', 'with $n=1{,}000$ i.i.d. observations', N),
    ('fixed_outcomes', 'of six independent outcomes drawn from a standard normal distribution', K),
    ('fixed_select_prob', 'picking the smallest $p$-value with probability 0.8 and picking a random one '
     'with probability 0.2', [1 - EPSILON, EPSILON]),
    ('fixed_m', 'We draw subsets of $m=31$ observations', M),
    ('fixed_budgets', r'for $B \in \{25,100, 400, 1000\}$', BUDGETS),
    ('fixed_replay_reject_100', r'4.2\% for the fixed design', main['replay'] / main['n']),
    ('fixed_agree_25', r'reaches the same rejection decision as the oracle correction in 96.2\% of simulations '
     'with $25$ replays', low['agree'] / low['n']),
    ('fixed_agree_1000', r'increasing to 99.8\% with $1{,}000$ replays', high['agree'] / high['n']),
    ('fixed_rmse_1000', 'falling to $0.019$ at $1{,}000$ replays', high['rmse']),
    ('fixed_width_1000', r'the mean width of the 95\% confidence intervals decreases to $0.069$', high['mean_width']),
    ('fixed_coverage_by_budget', "coverage of the known oracle correction also decreases with the number of "
     "replays, with some slight undercoverage at the larger replay budgets", coverage),
    ('fixed_median_lambda_by_budget', r'The median multiplier $\widehat \lambda$ remains stable around $3.5$', lambdas),
    ('fixed_lambda_rule', 'computed on simulations with analyses re-selected in both strata', printed_lambda),
    ('fixed_figure_design', 'the fixed design (400 datasets, 100 replays per stratum', [main['n'], FIGURE_BUDGET]),
])
