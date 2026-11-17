#!/usr/bin/env python3
"""Packets for the fact-label review of the masked Sonnet cohort's 2,800 extension replays (pilot/PLAN.md,
"Fact-label review of the full masked cohort"). No model calls.

Same arm content as the pilot's review (code/review_packets.py): the instruction, masked data-facts paragraph,
masked public context and target declaration are copied from the pilot's first packet and checked identical. The
candidates are the 2,800 replays at positions 201-3,000, in one random order (seed 2026110601), batches of 60,
blinded ids, 180 repeated pairs under new ids. The pilot's 200 keep their earlier labels.
Writes pilot/review/public/sonnet-masked-fact-ext/batch-NNN.json and pilot/review/private-map-ext.json (stays on
the Mac). Run from the folder root: python3 code/review_packets_ext.py
"""
import glob
import json
from pathlib import Path
import random
import secrets

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'pilot'
SEED, BATCH, REPEAT = 2026110601, 60, 180
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')

first = json.loads((PILOT / 'review/public/sonnet-masked-fact/batch-001.json').read_text())
target = json.load(open(PILOT / 'original/result.json'))['commitment']
assert first['target_declaration'] == {k: target[k] for k in FIELDS}

public = PILOT / 'review/public/sonnet-masked-fact-ext'
if public.exists():
    raise SystemExit('Packets already built; no rebuild')

recs = {}
for p in sorted(glob.glob(str(PILOT / 'node-results/replays-*.results.jsonl'))):
    for line in open(p):
        r = json.loads(line)
        prev = recs.get(r['case_id'])
        if prev is None or prev['status'] in ('unstarted_limit', 'interrupted_limit'):
            recs[r['case_id']] = r
pilot_ids = {v['case_id'] for v in json.load(open(PILOT / 'review/private-map.json'))['map'].values()}
replays = [r for r in recs.values() if r['status'] == 'complete' and r['position'] > 200]
assert len(replays) == 2800 and all(set(r['commitment']) == set(FIELDS) for r in replays)
assert not pilot_ids & {r['case_id'] for r in replays} and len(pilot_ids) == 200

public.mkdir(parents=True)
order = sorted(r['case_id'] for r in replays)
rng = random.Random(SEED)
rng.shuffle(order)
by_id = {r['case_id']: r['commitment'] for r in replays}
pairs = [(secrets.token_hex(32), c, False) for c in order]
pairs += [(secrets.token_hex(32), c, True) for c in rng.sample(order, REPEAT)]
private = {b: {'case_id': c, 'repeat': rep} for b, c, rep in pairs}
for i in range(0, len(pairs), BATCH):
    n = i // BATCH + 1
    batch = {'schema': 'card-rereview-batch-v1', 'arm': 'sonnet-masked-fact-ext', 'batch': n,
             'instruction': first['instruction'], 'public_context': first['public_context'],
             'target_declaration': first['target_declaration'],
             'pairs': [{'blinded_id': b, 'candidate_declaration': {k: by_id[c][k] for k in FIELDS}} for b, c, _ in pairs[i:i + BATCH]],
             'data_facts': first['data_facts']}
    (public / f'batch-{n:03d}.json').write_text(json.dumps(batch, ensure_ascii=False, indent=1))
path = PILOT / 'review/private-map-ext.json'
path.write_text(json.dumps({'seed': SEED, 'map': private}, indent=1))
path.chmod(0o600)
print(json.dumps({'batches': (len(pairs) + BATCH - 1) // BATCH, 'pairs': len(pairs), 'repeats': REPEAT}))
