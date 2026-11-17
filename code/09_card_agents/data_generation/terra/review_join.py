#!/usr/bin/env python3
"""Joins the finished declaration review to strata and raw verdicts
(REVIEW_PLAN.md). Reads review/out/results-*.jsonl, copied from Great Lakes, and
review/private-map.json. Writes processed/terra-reviewed.csv (one row per
classified replay), processed/review-disagreements.csv and
results/review-summary.json. No model calls.
Run from the run root: python3 code/review_join.py"""
from collections import Counter
import csv
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
results = [json.loads(line) for p in sorted((RUN / 'review/out').glob('results-*.jsonl'))
           for line in p.read_text().splitlines() if line.strip()]
done = [r for r in results if r['status'] == 'complete']
failed = [r['batch'] for r in results if r['status'] != 'complete']
batches = sorted(r['batch'] for r in done)
assert len(batches) == len(set(batches)), 'batch completed twice'
private = json.loads((RUN / 'review/private-map.json').read_text())['map']
labels = {x['blinded_id']: {**x, 'batch': r['batch'], 'account': r['account']} for r in done for x in r['labels']}
assert sum(len(r['labels']) for r in done) == len(labels), 'pair labelled twice'
by_case = {private[k]: v for k, v in labels.items()}

rows = []
for r in csv.DictReader(open(RUN / 'processed/terra-replays.csv')):
    if r['status'] != 'complete':
        continue
    x = by_case.get(r['case_id'])
    rows.append({'case_id': r['case_id'], 'stratum': r['stratum'], 'machine': r['machine'], 'bank_id': r['bank_id'],
                 'sandbox_refused': r['sandbox_refused'], 'raw_match': r['match'] == 'True',
                 'reviewed_label': x['label'] if x else 'not_reviewed', 'reviewed_reason': x['reason'] if x else '',
                 'raw_reason': r['reason'], 'batch': x['batch'] if x else '', 'account': x['account'] if x else ''})
with open(RUN / 'processed/terra-reviewed.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
changed = [r for r in rows if r['reviewed_label'] in ('match', 'nonmatch') and (r['reviewed_label'] == 'match') != r['raw_match']]
with open(RUN / 'processed/review-disagreements.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['case_id', 'stratum', 'raw_match', 'reviewed_label', 'reviewed_reason', 'raw_reason'], extrasaction='ignore')
    w.writeheader()
    w.writerows(changed)

usage = Counter()
for r in done:
    usage.update(r['usage'])
meters = {}
for r in sorted(done, key=lambda r: ((r.get('meter') or {}).get('primary') or {}).get('used_percent') or 0):
    meter = r.get('meter') or {}
    meters[r['account']] = {'used_percent': (meter.get('primary') or {}).get('used_percent'),
                            'credits': (meter.get('credits') or {}).get('balance')}
summary = {'pairs': len(rows), 'labelled': len(labels), 'batches_complete': len(done), 'batches_failed': failed,
           'attempts_over_one': sum(r['attempt'] > 1 for r in done), 'model': sorted({r['model'] for r in done}),
           'seconds_median': sorted(r['seconds'] for r in done)[len(done) // 2], 'usage': usage, 'meters': meters,
           'tool_items': Counter(t for r in done for t in r['tool_items']),
           'crosstab': {f"{s}, raw {m}": Counter(r['reviewed_label'] for r in rows if r['stratum'] == s and r['raw_match'] == m)
                        for s in ('tail', 'body') for m in (True, False)},
           'gained': sum(r['reviewed_label'] == 'match' for r in changed), 'lost': sum(r['reviewed_label'] == 'nonmatch' for r in changed)}
(RUN / 'results/review-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
