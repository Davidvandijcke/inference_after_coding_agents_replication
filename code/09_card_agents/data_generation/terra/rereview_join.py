#!/usr/bin/env python3
"""Joins the re-review of both Card cohorts (REVIEW_PLAN.md, amendment) to strata
and earlier labels. Reads rereview/out/results-*.jsonl, copied from Great Lakes,
with rereview/luna-map.json and review/private-map.json. Writes
processed/rereview-<arm>.csv (one row per pair) and results/rereview-summary.json.
No model calls. Run from the run root: python3 code/rereview_join.py"""
from collections import Counter
import csv
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ARMS = ('luna-current', 'luna-fact', 'terra-fact')
results = [json.loads(line) for p in sorted((RUN / 'rereview/out').glob('results-*.jsonl'))
           for line in p.read_text().splitlines() if line.strip()]
done = [r for r in results if r['status'] == 'complete']
units = [(r['arm'], r['batch']) for r in done]
assert len(units) == len(set(units)), 'unit completed twice'

luna_rows = json.loads((RUN.parent / 'card3000-independent-review-20261017/final3000-v1/reviewed-labels.json').read_text())['rows']
luna_raw = {r['job_id']: r for r in json.loads((RUN.parent / 'card3000-independent-review-20261017/final3000-v1/raw-labels.json').read_text())['rows']}
luna_map = json.loads((RUN / 'rereview/luna-map.json').read_text())
luna = {}
for r in luna_rows:
    first = 'match' if r['reviewed_match'] is True else 'nonmatch' if r['reviewed_match'] is False else 'unresolved'
    luna[r['job_id']] = {'case_id': r['job_id'], 'stratum': r['stratum'], 'raw_match': luna_raw[r['job_id']]['raw_match'], 'first_label': first}
terra_map = json.loads((RUN / 'review/private-map.json').read_text())['map']
terra = {r['case_id']: {'case_id': r['case_id'], 'stratum': r['stratum'], 'raw_match': r['raw_match'] == 'True', 'first_label': r['reviewed_label']}
         for r in csv.DictReader(open(RUN / 'processed/terra-reviewed.csv'))}

summary = {'batches_complete': Counter(r['arm'] for r in done), 'failed': [(r['arm'], r['batch']) for r in results if r['status'] != 'complete'],
           'attempts_over_one': sum(r['attempt'] > 1 for r in done), 'model': sorted({r['model'] for r in done})}
usage = Counter()
for r in done:
    usage.update(r['usage'])
summary['usage'] = usage
meters = {}
for r in sorted(done, key=lambda r: ((r.get('meter') or {}).get('primary') or {}).get('used_percent') or 0):
    meter = r.get('meter') or {}
    meters[r['account']] = {'used_percent': (meter.get('primary') or {}).get('used_percent'), 'credits': (meter.get('credits') or {}).get('balance')}
summary['meters'] = meters

for arm in ARMS:
    base = terra if arm.startswith('terra') else luna
    key = terra_map if arm.startswith('terra') else luna_map
    labels = {key[x['blinded_id']]: x for r in done if r['arm'] == arm for x in r['labels']}
    assert len(labels) == sum(len(r['labels']) for r in done if r['arm'] == arm), 'pair labelled twice'
    rows = [{**row, 'label': labels[c]['label'] if c in labels else 'not_reviewed', 'reason': labels[c]['reason'] if c in labels else ''}
            for c, row in base.items()]
    with open(RUN / f'processed/rereview-{arm}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary[arm] = {'pairs': len(rows), 'labelled': len(labels),
                    'labels_by_stratum': {s: Counter(r['label'] for r in rows if r['stratum'] == s) for s in ('tail', 'body')},
                    'against_first_review': {f'first {a}': Counter(r['label'] for r in rows if r['first_label'] == a) for a in ('match', 'nonmatch', 'unresolved')}}
(RUN / 'results/rereview-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
