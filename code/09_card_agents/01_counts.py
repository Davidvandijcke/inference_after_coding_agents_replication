#!/usr/bin/env python3
"""Matches and classified replays by stratum for the five cohorts of Table 8, on the reviewer's labels. Checks each
replay's stratum and subset p-value against its original's bank.

Inputs:  data/agent_runs/card_{luna,terra,sol,sonnet,sonnet_masked}/ (replays.csv.gz, original.json),
         data/intermediate/09_card_agents/*_bank_p.csv, data/intermediate/08_card/card-bank-p.csv
Output:  data/derived/09_card_agents/counts.csv
"""
import csv
import gzip
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'data' / 'agent_runs'
BANKS = {c: ROOT / 'data' / 'intermediate' / '09_card_agents' / f"{c.removeprefix('card_')}_bank_p.csv"
         for c in ('card_terra', 'card_sol', 'card_sonnet', 'card_sonnet_masked')}
BANKS['card_luna'] = ROOT / 'data' / 'intermediate' / '08_card' / 'card-bank-p.csv'
OUT = ROOT / 'data' / 'derived' / '09_card_agents'
COHORTS = ('card_luna', 'card_terra', 'card_sol', 'card_sonnet', 'card_sonnet_masked')
FIELDS = ['cohort', 'u', 'x1', 'b1', 'x0', 'b0', 'unresolved_tail', 'unresolved_body', 'unavailable_tail', 'unavailable_body']


def check(ok, message):
    if not ok:
        sys.exit(message)


def counts(rows, u, cohort):
    out = {'cohort': cohort, 'u': repr(u)}
    for s, k in (('tail', '1'), ('body', '0')):
        done = [r for r in rows if r['stratum'] == s and r['status'] == 'completed']
        classified = [r for r in done if r['review_label'] in ('match', 'nonmatch')]
        out[f'x{k}'] = sum(r['review_label'] == 'match' for r in classified)
        out[f'b{k}'] = len(classified)
        out[f'unresolved_{s}'] = len(done) - len(classified)
        out[f'unavailable_{s}'] = sum(r['stratum'] == s and r['status'] != 'completed' for r in rows)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    for cohort in COHORTS:
        rows = list(csv.DictReader(gzip.open(RUNS / cohort / 'replays.csv.gz', 'rt', newline='', encoding='utf-8')))
        u = json.load(open(RUNS / cohort / 'original.json', encoding='utf-8'))['p_value']
        check(all((float(r['bank_p']) <= u) == (r['stratum'] == 'tail') for r in rows), f'{cohort}: stratum and bank p disagree')
        bank = {r['bank_id']: float(r['p_value']) for r in csv.DictReader(open(BANKS[cohort], encoding='utf-8'))}
        check(all(math.isclose(float(r['bank_p']), bank[r['bank_id']], rel_tol=1e-12) for r in rows),
              f'{cohort}: subset p differs from the bank')
        table.append(counts(rows, u, cohort))
    with open(OUT / 'counts.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows(table)
    for r in table:
        print(f"{r['cohort']:<19} tail {r['x1']:>5}/{r['b1']:<5} body {r['x0']:>5}/{r['b0']:<5}"
              f" unresolved {r['unresolved_tail']}+{r['unresolved_body']} unavailable {r['unavailable_tail']}+{r['unavailable_body']}")


if __name__ == '__main__':
    main()
