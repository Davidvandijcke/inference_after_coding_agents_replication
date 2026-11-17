#!/usr/bin/env python3
"""Further review batches for Sol Card replays completed after the first 831 (REVIEW_PLAN.md). No model calls.

Same arm design as sol_rereview_packets.py: the same instruction, public context, target and data facts,
batches of 60 in one random order, blinded ids of 32 random bytes, a number of repeated pairs under new
ids. Writes rereview/public/<ARM>/batch-NNN.json and merges the new ids into rereview/private-map.json.
Run from the run root: python3 code/sol_rereview_packets_more.py ARM REPLAYS.jsonl REPEATS SEED
"""
import json
from pathlib import Path
import random
import secrets
import sys

RUN = Path(__file__).resolve().parents[1]
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
arm, replays_file, repeats, seed = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
BATCH = 60

first = json.loads(sorted((RUN / 'rereview/public/sol-fact').glob('batch-*.json'))[0].read_text())
public = RUN / 'rereview/public' / arm
if public.exists():
    raise SystemExit('Packets already built; no rebuild')
public.mkdir(parents=True)
private_path = RUN / 'rereview/private-map.json'
private = json.loads(private_path.read_text())
known = {v['case_id'] for v in private['map'].values()}
replays = [r for r in map(json.loads, open(replays_file)) if r.get('status') == 'complete']
assert all(set(r['commitment']) == set(FIELDS) for r in replays)
new = sorted(r['case_id'] for r in replays if r['case_id'] not in known)
assert len(new) == len(replays), 'a replay in this file was already reviewed'
rng = random.Random(seed)
rng.shuffle(new)
by_id = {r['case_id']: r['commitment'] for r in replays}
pairs = [(secrets.token_hex(32), c, False) for c in new] + [(secrets.token_hex(32), c, True) for c in rng.sample(new, repeats)]
for b, c, rep in pairs:
    private['map'][b] = {'case_id': c, 'repeat': rep, 'arm': arm}
for i in range(0, len(pairs), BATCH):
    n = i // BATCH + 1
    batch = {'schema': 'card-rereview-batch-v1', 'arm': arm, 'batch': n, 'instruction': first['instruction'],
             'public_context': first['public_context'], 'target_declaration': first['target_declaration'],
             'pairs': [{'blinded_id': b, 'candidate_declaration': {k: by_id[c][k] for k in FIELDS}} for b, c, _ in pairs[i:i + BATCH]],
             'data_facts': first['data_facts']}
    (public / f'batch-{n:03d}.json').write_text(json.dumps(batch, ensure_ascii=False, indent=1))
private.setdefault('arms', {})[arm] = {'seed': seed, 'replays': replays_file.name, 'pairs': len(new), 'repeats': repeats}
private_path.write_text(json.dumps(private, indent=1))
private_path.chmod(0o600)
print(json.dumps({'arm': arm, 'batches': (len(pairs) + BATCH - 1) // BATCH, 'pairs': len(new), 'repeats': repeats}))
