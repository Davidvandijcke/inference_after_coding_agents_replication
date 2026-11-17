#!/usr/bin/env python3
"""Packets for the fact-label review of the Sol Card cohort (REVIEW_PLAN.md). No model calls.

One arm, sol-fact: every replay's committed declaration against Sol's original declaration, with the
Terra re-review's instruction, rule and data-facts paragraph, in batches of 60 in one random order.
Two further batches repeat 120 pairs under new blinded ids. Writes rereview/public/sol-fact/batch-NNN.json
and rereview/private-map.json (blinded id to case id, with repeat flags), which stays on the Mac.
Run from the run root: python3 code/sol_rereview_packets.py
"""
import hashlib
import json
from pathlib import Path
import random
import re
import secrets

RUN = Path(__file__).resolve().parents[1]
TERRA = RUN.parent / 'card-terra-subscription-pilot-20261029'
SEED, BATCH, REPEAT = 2026103101, 60, 120
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')

plan = (TERRA / 'REVIEW_PLAN.md').read_text()
FACTS = '\n'.join(line[2:] if line.startswith('> ') else '' for line in plan.splitlines() if line.startswith('>'))
assert FACTS.startswith('Facts about these data') and FACTS.endswith('remain different restrictions.')
FACTS = re.sub(r'(?<!\n)\n(?!\n)', ' ', FACTS)
template = json.loads(next(sorted((TERRA / 'rereview/public/terra-fact').glob('batch-*.json')).__iter__()).read_text())
assert template['data_facts'] == FACTS

public = RUN / 'rereview/public/sol-fact'
if public.exists():
    raise SystemExit('Packets already built; no rebuild')
public.mkdir(parents=True)

context = []
for name in ('DESIGN.md', 'CODEBOOK.md'):
    text = (RUN / 'inputs' / name).read_text()
    context.append({'name': name, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text})
target = json.load(open(RUN / 'runs/original/result.json'))['commitment']
assert set(target) == set(FIELDS)
replays = [json.loads(line) for line in open(RUN / 'processed/replays.jsonl')]
assert len(replays) == 831 and all(set(r['commitment']) == set(FIELDS) for r in replays)

order = sorted(r['case_id'] for r in replays)
rng = random.Random(SEED)
rng.shuffle(order)
by_id = {r['case_id']: r['commitment'] for r in replays}
pairs = [(secrets.token_hex(32), c, False) for c in order]
pairs += [(secrets.token_hex(32), c, True) for c in rng.sample(order, REPEAT)]
private = {b: {'case_id': c, 'repeat': rep} for b, c, rep in pairs}
for i in range(0, len(pairs), BATCH):
    n = i // BATCH + 1
    batch = {'schema': 'card-rereview-batch-v1', 'arm': 'sol-fact', 'batch': n, 'instruction': template['instruction'],
             'public_context': context, 'target_declaration': {k: target[k] for k in FIELDS},
             'pairs': [{'blinded_id': b, 'candidate_declaration': {k: by_id[c][k] for k in FIELDS}} for b, c, _ in pairs[i:i + BATCH]],
             'data_facts': FACTS}
    (public / f'batch-{n:03d}.json').write_text(json.dumps(batch, ensure_ascii=False, indent=1))
path = RUN / 'rereview/private-map.json'
path.write_text(json.dumps({'seed': SEED, 'map': private}, indent=1))
path.chmod(0o600)
print(json.dumps({'batches': (len(pairs) + BATCH - 1) // BATCH, 'pairs': len(pairs), 'repeats': REPEAT,
                  'public_context': [(c['name'], c['sha256'][:10]) for c in context]}))
