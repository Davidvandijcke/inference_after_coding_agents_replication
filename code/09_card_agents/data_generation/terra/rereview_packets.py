#!/usr/bin/env python3
"""Packets for the re-review of both Card cohorts (REVIEW_PLAN.md, amendment).
Writes rereview/public/<arm>/batch-NNN.json and rereview/luna-map.json, which
maps luna's blinded IDs to jobs and stays on the Mac. Terra's arm reuses the
first review's batches. No model calls.
Run from the run root: python3 code/rereview_packets.py"""
import glob
import json
from pathlib import Path
import random
import re

RUN = Path(__file__).resolve().parents[1]
OUT = RUN.parent
SEED, BATCH = 2026103002, 60
plan = (RUN / 'REVIEW_PLAN.md').read_text()
FACTS = '\n'.join(line[2:] if line.startswith('> ') else '' for line in plan.splitlines() if line.startswith('>'))
assert FACTS.startswith('Facts about these data') and FACTS.endswith('remain different restrictions.')
FACTS = re.sub(r'(?<!\n)\n(?!\n)', ' ', FACTS)
public = RUN / 'rereview/public'
if public.exists():
    raise SystemExit('Packets already built; no rebuild')

first = sorted((RUN / 'review/public').glob('batch-*.json'))
(public / 'terra-fact').mkdir(parents=True)
for path in first:
    batch = json.loads(path.read_text())
    batch['data_facts'] = FACTS
    (public / 'terra-fact' / path.name).write_text(json.dumps(batch, ensure_ascii=False, indent=1))

template = json.loads(first[0].read_text())
pairs = {}
for f in sorted(glob.glob(str(OUT / 'card3000-bulk-execution-20261017/seal-*/review/public/batch-*.json'))):
    for p in json.loads(open(f).read())['pairs']:
        pairs.setdefault(p['blinded_id'], p)
assert len(pairs) == 3000
rows = json.loads((OUT / 'card3000-independent-review-20261017/final3000-v1/reviewed-labels.json').read_text())['rows']
job = {r['review_provenance'][0]['blinded_id']: r['job_id'] for r in rows}
assert set(job) == set(pairs)
one = next(iter(pairs.values()))
assert one['instruction'] == template['instruction']
assert [c['sha256'] for c in one['public_context']] == [c['sha256'] for c in template['public_context']]
order = sorted(pairs)
random.Random(SEED).shuffle(order)
for arm, facts in (('luna-current', None), ('luna-fact', FACTS)):
    (public / arm).mkdir()
    for i in range(0, len(order), BATCH):
        n = i // BATCH + 1
        batch = {'schema': 'card-rereview-batch-v1', 'batch': n, 'instruction': one['instruction'],
                 'public_context': template['public_context'], 'target_declaration': one['target_declaration'],
                 'pairs': [{'blinded_id': k, 'candidate_declaration': pairs[k]['candidate_declaration']} for k in order[i:i + BATCH]]}
        if facts:
            batch['data_facts'] = facts
        (public / arm / f'batch-{n:03d}.json').write_text(json.dumps(batch, ensure_ascii=False, indent=1))
(RUN / 'rereview/luna-map.json').write_text(json.dumps(job))
print({p.name: len(list(p.glob('*.json'))) for p in sorted(public.iterdir())})
print(FACTS)
