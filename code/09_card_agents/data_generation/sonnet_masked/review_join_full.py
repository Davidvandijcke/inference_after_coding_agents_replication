#!/usr/bin/env python3
"""Joins the fact-label review of the full masked Sonnet cohort (pilot/PLAN.md, "Fact-label review of the full
masked cohort") to strata and the replays' own verdicts. No model calls.

Reads pilot/review/out/results-*.jsonl (copied from Great Lakes; both arms, sonnet-masked-fact for the pilot's 200
and sonnet-masked-fact-ext for the 2,800 added replays), the two private maps and the node results. Writes
pilot/processed/sonnet-masked-fact-3000.csv, pilot/processed/sonnet-masked-fact-3000-repeats.csv and
pilot/results/review-summary-3000.json. Run from the folder root: python3 code/review_join_full.py
"""
from collections import Counter
import csv
import glob
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'pilot'
MAPS = {'sonnet-masked-fact': 'private-map.json', 'sonnet-masked-fact-ext': 'private-map-ext.json'}
results = [json.loads(line) for p in sorted((P / 'review/out').glob('results-*.jsonl')) for line in p.read_text().splitlines() if line.strip()]
done = [r for r in results if r['status'] == 'complete']
assert len({(r['arm'], r['batch']) for r in done}) == len(done), 'batch completed twice'
private = {arm: json.load(open(P / 'review' / f))['map'] for arm, f in MAPS.items()}
recs = {}
for p in sorted(glob.glob(str(P / 'node-results/replays-*.results.jsonl'))):
    for line in open(p):
        r = json.loads(line)
        prev = recs.get(r['case_id'])
        if prev is None or prev['status'] in ('unstarted_limit', 'interrupted_limit'):
            recs[r['case_id']] = r
replays = [r for r in recs.values() if r['status'] == 'complete']
assert len(replays) == 3000
stratum = {r['case_id']: r['stratum'] for r in replays}
part = {r['case_id']: 'pilot' if r['position'] <= 200 else 'ext' for r in replays}
raw = {r['case_id']: bool((r.get('verdict') or {}).get('model_null_match')) for r in replays}

first, repeats = {}, []
for r in done:
    for x in r['labels']:
        k = private[r['arm']][x['blinded_id']]
        c = k['case_id']
        row = {'case_id': c, 'part': part[c], 'stratum': stratum[c], 'raw_match': raw[c], 'label': x['label'], 'reason': x['reason']}
        (repeats.append(row) if k['repeat'] else first.__setitem__(c, row))
rows = [first.get(c, {'case_id': c, 'part': part[c], 'stratum': stratum[c], 'raw_match': raw[c], 'label': 'not_reviewed', 'reason': ''}) for c in sorted(raw)]
for name, data in (('sonnet-masked-fact-3000.csv', rows), ('sonnet-masked-fact-3000-repeats.csv', repeats)):
    with open(P / 'processed' / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case_id', 'part', 'stratum', 'raw_match', 'label', 'reason'])
        w.writeheader()
        w.writerows(data)
agree = sum(first[r['case_id']]['label'] == r['label'] for r in repeats if r['case_id'] in first)
usage = Counter()
for r in done:
    usage.update(r['usage'])
summary = {'batches_complete': dict(Counter(r['arm'] for r in done)), 'failed': [(r['arm'], r['batch']) for r in results if r['status'] != 'complete'],
           'attempts_over_one': sum(r['attempt'] > 1 for r in done), 'model': sorted({r['model'] for r in done}), 'usage': dict(usage),
           'labelled': len(first), 'labels_by_stratum': {s: dict(Counter(r['label'] for r in rows if r['stratum'] == s)) for s in ('tail', 'body')},
           'labels_by_part': {p: dict(Counter(r['label'] for r in rows if r['part'] == p)) for p in ('pilot', 'ext')},
           'against_raw': {f'raw {a}': dict(Counter(r['label'] for r in rows if r['raw_match'] == a)) for a in (True, False)},
           'repeats': {'pairs': len(repeats), 'agree': agree}}
(P / 'results/review-summary-3000.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
