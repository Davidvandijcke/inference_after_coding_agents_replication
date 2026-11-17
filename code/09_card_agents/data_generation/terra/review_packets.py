#!/usr/bin/env python3
"""Blinded declaration-review packets for the Terra cohort (REVIEW_PLAN.md).
Writes review/public/batch-NNN.json, which go to Great Lakes, and
review/private-map.json, which stays on the Mac. No model calls.
Run from the run root: python3 code/review_packets.py"""
import csv
import hashlib
import json
from pathlib import Path
import random
import secrets

RUN = Path(__file__).resolve().parents[1]
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
SEED, BATCH = 2026103001, 60
# Verbatim from the luna review's packets (card-fixed3000-blind-declaration-pair-v1).
INSTRUCTION = ('Compare the declared statistical model, null, outcome, population and sample restrictions. '
               'Ignore estimator, inference procedure, code and presentation differences. Use supplied variable '
               'definitions for genuine shorthand, preserve explicit conflicts, and mark unresolved rather than '
               'infer from code, numerical behavior or abandoned alternatives.')

public, private = RUN / 'review/public', RUN / 'review/private-map.json'
if private.exists():
    raise SystemExit('Packets already built; no rebuild')
public.mkdir(parents=True)
target = json.loads((RUN / 'runs/original/result.json').read_text())['commitment']
rows = [r for r in csv.DictReader(open(RUN / 'processed/terra-replays.csv')) if r['status'] == 'complete']
salt = secrets.token_hex(32)
pairs = [{'blinded_id': hashlib.sha256((salt + r['case_id']).encode()).hexdigest(),
          'case_id': r['case_id'],
          'candidate_declaration': {k: r[f'commit_{k}'] for k in FIELDS}} for r in rows]
random.Random(SEED).shuffle(pairs)
context = [{'name': n, 'sha256': hashlib.sha256((RUN / 'inputs' / n).read_bytes()).hexdigest(),
            'text': (RUN / 'inputs' / n).read_text()} for n in ('DESIGN.md', 'CODEBOOK.md')]
for i in range(0, len(pairs), BATCH):
    n = i // BATCH + 1
    batch = {'schema': 'terra-blind-declaration-batch-v1', 'batch': n, 'instruction': INSTRUCTION,
             'public_context': context, 'target_declaration': target,
             'pairs': [{'blinded_id': p['blinded_id'], 'candidate_declaration': p['candidate_declaration']}
                       for p in pairs[i:i + BATCH]]}
    (public / f'batch-{n:03d}.json').write_text(json.dumps(batch, ensure_ascii=False, indent=1))
private.write_text(json.dumps({'salt': salt, 'map': {p['blinded_id']: p['case_id'] for p in pairs}}))
private.chmod(0o600)
print(len(pairs), 'pairs in', (len(pairs) + BATCH - 1) // BATCH, 'batches')
