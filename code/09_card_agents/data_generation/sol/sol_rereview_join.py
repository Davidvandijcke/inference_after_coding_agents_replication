#!/usr/bin/env python3
"""Joins the Sol Card fact-label review to strata and Sol's own verdicts. No model calls.

Reads rereview/out/results-*.jsonl (copied from Great Lakes), rereview/private-map.json, main-roster.json
and processed/replays.jsonl. Writes processed/sol-fact.csv (one row per replay: case_id, stratum,
raw_match, label, reason), processed/sol-fact-repeats.csv and results/rereview-summary.json.
Run from the run root: python3 code/sol_rereview_join.py
"""
from collections import Counter
import csv
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
results = [json.loads(line) for p in sorted((RUN / 'rereview/out').glob('results-*.jsonl')) for line in p.read_text().splitlines() if line.strip()]
done = [r for r in results if r['status'] == 'complete']
assert len({(r.get('arm'), r['batch']) for r in done}) == len(done), 'batch completed twice'
private = json.load(open(RUN / 'rereview/private-map.json'))['map']
stratum = {c['case_id']: c['stratum'] for c in json.load(open(RUN / 'main-roster.json'))['cases']}
raw = {r['case_id']: bool((r.get('verdict') or {}).get('model_null_match')) for p in sorted((RUN / 'processed').glob('replays*.jsonl'))
       for r in map(json.loads, open(p)) if r.get('status') == 'complete'}

first, repeats = {}, []
for r in done:
    for x in r['labels']:
        k = private[x['blinded_id']]
        row = {'case_id': k['case_id'], 'stratum': stratum[k['case_id']], 'raw_match': raw[k['case_id']], 'label': x['label'], 'reason': x['reason']}
        (repeats.append(row) if k['repeat'] else first.__setitem__(k['case_id'], row))
rows = [first.get(c, {'case_id': c, 'stratum': stratum[c], 'raw_match': raw[c], 'label': 'not_reviewed', 'reason': ''}) for c in sorted(raw)]
(RUN / 'processed').mkdir(exist_ok=True)
for name, data in (('sol-fact.csv', rows), ('sol-fact-repeats.csv', repeats)):
    with open(RUN / 'processed' / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case_id', 'stratum', 'raw_match', 'label', 'reason'])
        w.writeheader()
        w.writerows(data)
agree = sum(first[r['case_id']]['label'] == r['label'] for r in repeats if r['case_id'] in first)
usage = Counter()
for r in done:
    usage.update(r['usage'])
summary = {'batches_complete': len(done), 'failed': [r['batch'] for r in results if r['status'] != 'complete'],
           'attempts_over_one': sum(r['attempt'] > 1 for r in done), 'model': sorted({r['model'] for r in done}), 'usage': dict(usage),
           'meters': {r['account']: ((r.get('meter') or {}).get('primary') or {}).get('used_percent') for r in done},
           'labelled': len(first), 'labels_by_stratum': {s: dict(Counter(r['label'] for r in rows if r['stratum'] == s)) for s in ('tail', 'body')},
           'against_raw': {f'raw {a}': dict(Counter(r['label'] for r in rows if r['raw_match'] == a)) for a in (True, False)},
           'repeats': {'pairs': len(repeats), 'agree': agree}}
(RUN / 'results').mkdir(exist_ok=True)
(RUN / 'results/rereview-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
