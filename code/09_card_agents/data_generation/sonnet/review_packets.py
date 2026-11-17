#!/usr/bin/env python3
"""Packets for the fact-label review of the Sonnet Card cohort (MAIN_PLAN.md, "Fact-label review"). No model calls.

One arm, sonnet-fact: every replay's committed declaration against Sonnet's original declaration, with the
instruction, public context and data-facts paragraph of the Sol cohort's sol-fact arm (the paper's rule, as for
Luna, Terra and Sol), in batches of 60 in one random order. Further batches repeat 200 pairs under new blinded ids.
Writes review/public/sonnet-fact/batch-NNN.json and review/private-map.json (blinded id to case id), which stays
on the Mac. Run from the run root: python3 code/review_packets.py
"""
import hashlib
import json
from pathlib import Path
import random
import secrets

RUN = Path(__file__).resolve().parents[1]
SOL = RUN.parent / 'card-sol-subscription-pilot-20261029'
SEED, BATCH, REPEAT = 2026110501, 60, 200
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')

first = json.loads(sorted((SOL / 'rereview/public/sol-fact').glob('batch-*.json'))[0].read_text())
for c in first['public_context']:  # the Sonnet cohort's packet is the Sol cohort's
    assert hashlib.sha256((RUN / 'inputs' / c['name']).read_bytes()).hexdigest() == c['sha256'], c['name']

public = RUN / 'review/public/sonnet-fact'
if public.exists():
    raise SystemExit('Packets already built; no rebuild')
public.mkdir(parents=True)

target = json.load(open(RUN / 'processed/original-result.json'))['commitment']
assert set(target) == set(FIELDS)
replays = [json.loads(line) for line in open(RUN / 'processed/replays.jsonl')]
replays = [r for r in replays if r.get('status') == 'complete']
assert len(replays) == 3000 and all(set(r['commitment']) == set(FIELDS) for r in replays)

order = sorted(r['case_id'] for r in replays)
rng = random.Random(SEED)
rng.shuffle(order)
by_id = {r['case_id']: r['commitment'] for r in replays}
pairs = [(secrets.token_hex(32), c, False) for c in order]
pairs += [(secrets.token_hex(32), c, True) for c in rng.sample(order, REPEAT)]
private = {b: {'case_id': c, 'repeat': rep} for b, c, rep in pairs}
for i in range(0, len(pairs), BATCH):
    n = i // BATCH + 1
    batch = {'schema': 'card-rereview-batch-v1', 'arm': 'sonnet-fact', 'batch': n, 'instruction': first['instruction'],
             'public_context': first['public_context'], 'target_declaration': {k: target[k] for k in FIELDS},
             'pairs': [{'blinded_id': b, 'candidate_declaration': {k: by_id[c][k] for k in FIELDS}} for b, c, _ in pairs[i:i + BATCH]],
             'data_facts': first['data_facts']}
    (public / f'batch-{n:03d}.json').write_text(json.dumps(batch, ensure_ascii=False, indent=1))
path = RUN / 'review/private-map.json'
path.write_text(json.dumps({'seed': SEED, 'map': private}, indent=1))
path.chmod(0o600)
print(json.dumps({'batches': (len(pairs) + BATCH - 1) // BATCH, 'pairs': len(pairs), 'repeats': REPEAT}))
