#!/usr/bin/env python3
"""Table 8 (one row per cohort, on the reviewer's labels) and the module's in-text numbers.

Inputs:  data/derived/09_card_agents/estimates.csv, keyword_rates.json, data/agent_runs/card_sonnet/original.json
Outputs: paper/tabs/card_agents.tex, results/numbers/09_card_agents.json
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code' / 'lib'))
from paper_numbers import write_numbers  # noqa: E402

DERIVED = ROOT / 'data' / 'derived' / '09_card_agents'
ROWS = (('card_luna', 'GPT-6-Luna'), ('card_terra', 'GPT-6.1-Terra'), ('card_sol', 'GPT-6.1-Sol'),
        ('card_sonnet', 'Claude Sonnet 5.5'), ('card_sonnet_masked', r'\quad masked data'))
QUOTE = ('I did not lead with the most statistically compelling result, because picking the smallest p-value '
         'from about 30 looks would overstate the evidence.')


def matches(x, b):
    return f'{x:,}/{b:,} ({100 * x / b:.1f}\\%)'


def table(est):
    lines = [r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
             r' & Original & Tail & Body & Multiplier & Corrected \\',
             r'Agent & $p$-value & matches & matches & $\lambda$ & $p$-value \\', r'\midrule']
    for cohort, label in ROWS:
        r = est[cohort]
        lines.append(f"{label} & {r['u']:.3f} & {matches(r['x1'], r['b1'])} & {matches(r['x0'], r['b0'])} & "
                     f"{r['multiplier']:.2f} ($\\geq$ {r['multiplier_lower_one95']:.2f}) & "
                     f"{r['p_corrected']:.3f} ($\\leq$ {r['upper_one95']:.3f}) \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def numbers(est, kw, report):
    terra, sol, sonnet, masked = est['card_terra'], est['card_sol'], est['card_sonnet'], est['card_sonnet_masked']
    share = lambda c, k: kw[c][k + '_share']
    return [
        ('card_sol_lambda_intro', 'GPT-6.1-Sol is nearly 2 times more likely to report its chosen analysis when it looks '
         'statistically favorable', sol['multiplier']),
        ('card_terra_lambda', r'with $\widehat \lambda = 1.79, 1.98$', terra['multiplier']),
        ('card_sol_lambda', r'with $\widehat \lambda = 1.79, 1.98$', sol['multiplier']),
        ('card_terra_p', r'$\widehat p^{\mathrm{sel}} = 0.051, 0.053$, respectively', terra['p_corrected']),
        ('card_sol_p', r'$\widehat p^{\mathrm{sel}} = 0.051, 0.053$, respectively', sol['p_corrected']),
        ('card_sonnet_lambda', 'the estimated multipliers for Claude Sonnet 5.5 are almost exactly equal to 1', sonnet['multiplier']),
        ('card_sonnet_quote', "``" + QUOTE + "''", QUOTE if QUOTE in report else None),
        ('card_sonnet_names_text', r'explicitly mentioning the paper in its logs 18.7\% of the time', share('card_sonnet', 'name')),
        ('card_sol_names_text', "compared to none of GPT-6.1-Sol's replays", kw['card_sol']['name']),
        ('card_sonnet_masked_lambda', r"On the masked data, Claude's estimated $\lambda$ is slightly higher ($1.01$)",
         masked['multiplier']),
        ('card_sonnet_masked_not_significant', 'but remains insignificantly different from $1$', masked['multiplier_lower_one95']),
        ('card_sonnet_declines', r'in its visible reasoning or messages in 97.1\% of replays', share('card_sonnet', 'decline')),
        ('card_sonnet_masked_declines', r'(masked data: 97.9\%)', share('card_sonnet_masked', 'decline')),
        ('card_luna_declines', "against 3 of GPT-6-Luna's 3,000", kw['card_luna']['decline']),
        ('card_sol_declines', "3,000 and none of GPT-6.1-Sol's", kw['card_sol']['decline']),
        ('card_sonnet_names', r'names Card and Krueger somewhere in the session in 18.7\%', share('card_sonnet', 'name')),
        ('card_sonnet_masked_names', r'(masked data: 10.6\%)', share('card_sonnet_masked', 'name')),
        ('card_luna_names', "against 1 of GPT-6-Luna's", kw['card_luna']['name']),
        ('card_sol_names', "against 1 of GPT-6-Luna's and none of GPT-6.1-Sol's", kw['card_sol']['name']),
    ]


def main():
    est = {}
    for r in csv.DictReader(open(DERIVED / 'estimates.csv', encoding='utf-8')):
        est[r['cohort']] = {k: r[k] if k == 'cohort' else int(r[k]) if k in ('x1', 'b1', 'x0', 'b0') else float(r[k])
                            for k in r}
    kw = json.load(open(DERIVED / 'keyword_rates.json', encoding='utf-8'))
    report = json.load(open(ROOT / 'data' / 'agent_runs' / 'card_sonnet' / 'original.json', encoding='utf-8'))['final_report']
    (ROOT / 'paper' / 'tabs').mkdir(parents=True, exist_ok=True)
    (ROOT / 'paper' / 'tabs' / 'card_agents.tex').write_text(table(est), encoding='utf-8')
    write_numbers('09_card_agents', numbers(est, kw, report))
    print((ROOT / 'paper' / 'tabs' / 'card_agents.tex').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
