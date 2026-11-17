#!/usr/bin/env python3
"""Packets for the fact-label review of the masked Sonnet Card pilot (pilot/PLAN.md, "Fact-label review"). No model calls.

One arm, sonnet-masked-fact: each of the pilot's 200 replays' committed declarations against the masked original's
declaration. Instruction and data-facts paragraph from the Sol cohort's sol-fact arm (the paper's rule), with the
facts' four column names given their masked names (status2, empft2, emppt2, nmgrs2 -> status_w2, fulltime_w2,
parttime_w2, managers_w2); every fact was checked on the masked file (pilot/PLAN.md). Public context: the masked
packet's DESIGN.md and CODEBOOK.md (inputs-v2b), which the agents saw. Batches of 60 in one random order, blinded ids,
20 repeated pairs. Writes pilot/review/public/sonnet-masked-fact/batch-NNN.json and pilot/review/private-map.json
(stays on the Mac). Run from the folder root: python3 code/review_packets.py
"""
import glob
import hashlib
import json
from pathlib import Path
import random
import secrets

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'pilot'
SOL = ROOT.parent / 'card-sol-subscription-pilot-20261029'
SEED, BATCH, REPEAT = 2026110502, 60, 20
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
RENAME = (('status2', 'status_w2'), ('empft2', 'fulltime_w2'), ('emppt2', 'parttime_w2'), ('nmgrs2', 'managers_w2'))

first = json.loads(sorted((SOL / 'rereview/public/sol-fact').glob('batch-*.json'))[0].read_text())
facts = first['data_facts']
for old, new in RENAME:
    assert old in facts
    facts = facts.replace(old, new)
context = [{'name': n, 'sha256': hashlib.sha256((ROOT / 'inputs-v2b' / n).read_bytes()).hexdigest(),
            'text': (ROOT / 'inputs-v2b' / n).read_text()} for n in ('DESIGN.md', 'CODEBOOK.md')]

public = PILOT / 'review/public/sonnet-masked-fact'
if public.exists():
    raise SystemExit('Packets already built; no rebuild')
public.mkdir(parents=True)

target = json.load(open(PILOT / 'original/result.json'))['commitment']
assert set(target) == set(FIELDS)
recs = {}
for p in sorted(glob.glob(str(PILOT / 'node-results/replays-*.results.jsonl'))):
    for line in open(p):
        r = json.loads(line)
        prev = recs.get(r['case_id'])
        if prev is None or prev['status'] in ('unstarted_limit', 'interrupted_limit'):
            recs[r['case_id']] = r
replays = [r for r in recs.values() if r['status'] == 'complete']
assert len(replays) == 200 and all(set(r['commitment']) == set(FIELDS) for r in replays)

order = sorted(r['case_id'] for r in replays)
rng = random.Random(SEED)
rng.shuffle(order)
by_id = {r['case_id']: r['commitment'] for r in replays}
pairs = [(secrets.token_hex(32), c, False) for c in order]
pairs += [(secrets.token_hex(32), c, True) for c in rng.sample(order, REPEAT)]
private = {b: {'case_id': c, 'repeat': rep} for b, c, rep in pairs}
for i in range(0, len(pairs), BATCH):
    n = i // BATCH + 1
    batch = {'schema': 'card-rereview-batch-v1', 'arm': 'sonnet-masked-fact', 'batch': n, 'instruction': first['instruction'],
             'public_context': context, 'target_declaration': {k: target[k] for k in FIELDS},
             'pairs': [{'blinded_id': b, 'candidate_declaration': {k: by_id[c][k] for k in FIELDS}} for b, c, _ in pairs[i:i + BATCH]],
             'data_facts': facts}
    (public / f'batch-{n:03d}.json').write_text(json.dumps(batch, ensure_ascii=False, indent=1))
path = PILOT / 'review/private-map.json'
path.write_text(json.dumps({'seed': SEED, 'map': private}, indent=1))
path.chmod(0o600)
print(json.dumps({'batches': (len(pairs) + BATCH - 1) // BATCH, 'pairs': len(pairs), 'repeats': REPEAT, 'facts': facts[:160]}))
