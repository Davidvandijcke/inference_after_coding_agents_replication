"""Replays, classified replays and matches per original, in the tail and in the body.

A replay is classified when its comparison returned a verdict, and a match when that verdict says it selected
the original's model and null. Of several records of one session the last classified one counts. Also checks
that every session received the data its record names and that each replay's stratum follows the reference bank.

Inputs:  data/agent_runs/open_ended_luna/, data/raw/pathways/, data/intermediate/02_open_ended/bank_pvalues.csv.gz
Outputs: data/derived/02_open_ended/replay_counts.csv, design.json
"""
import csv
import gzip
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / 'data' / 'agent_runs' / 'open_ended_luna'
BANK = ROOT / 'data' / 'intermediate' / '02_open_ended' / 'bank_pvalues.csv.gz'
OUT = ROOT / 'data' / 'derived' / '02_open_ended'
M = 200  # subsample size

with gzip.open(COHORT / 'sessions.jsonl.gz', 'rt', encoding='utf-8') as f:
    sessions = [json.loads(line) for line in f]
replays = [r for r in sessions if r['role'] == 'replay']
originals = list(csv.DictReader(open(COHORT / 'originals.csv', encoding='utf-8')))
seeds = {int(r['slot']): int(r['seed']) for r in originals}
u = {int(r['slot']): float(r['u']) for r in originals if r['u']}
datasets = {}
for s in seeds:
    with open(ROOT / 'data' / 'raw' / 'pathways' / f'main-{s:02d}.csv', 'rb') as f:
        raw = f.read()
    datasets[s] = (hashlib.sha256(raw).hexdigest(), list(csv.reader(io.StringIO(raw.decode('utf-8'), newline=''))))
(N,) = {len(rows) - 1 for _, rows in datasets.values()}


def served(slot, j):
    """SHA-256 of the data.csv a replay of `slot` on bank subsample `j` received."""
    rows = datasets[slot][1]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(rows[0])
    w.writerows(rows[1 + i] for i in random.Random(f'{seeds[slot]}-bank-{j}').sample(range(N), M))
    return hashlib.sha256(buf.getvalue().encode('utf-8')).hexdigest()


assert all(r['data_sha256'] == datasets[r['slot']][0] for r in sessions if r['role'] == 'original')
assert all(r['data_sha256'] == served(r['slot'], r['bank_index']) for r in replays)

drawn = {(r['slot'], r['bank_index']) for r in replays}
with gzip.open(BANK, 'rt', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    assert next(reader) == ['slot', 'bank_index', 'p_value']
    bank = {key: float(p) if p else None for slot, j, p in reader if (key := (int(slot), int(j))) in drawn}
assert all(r['ref_p'] == bank[r['slot'], r['bank_index']] for r in replays)
assert all((r['ref_p'] <= u[r['slot']]) == (r['stratum'] == 'tail') for r in replays)

chosen = {}
records = [r for r in replays if r['sink'] == 'results']
for r in sorted(records, key=lambda r: (r['file'], r['line'])):
    prev = chosen.get(r['session_id'])
    if prev is None or r['model_null_match'] is not None or prev['model_null_match'] is None:
        chosen[r['session_id']] = r

n = Counter()
for r in chosen.values():
    key = (r['slot'], r['component'], r['stratum'])
    n[key + ('attempted',)] += 1
    n[key + ('classified',)] += r['model_null_match'] is not None
    n[key + ('match',)] += r['model_null_match'] is True

rows = []
for slot, comp in sorted({(s, c) for s, c, _, _ in n}):
    x = {st: [n[(slot, comp, st, k)] for k in ('match', 'classified', 'attempted')] for st in ('tail', 'body')}
    (x1, b1, a1), (x0, b0, a0) = x['tail'], x['body']
    rows.append([slot, comp, a1 + a0, x1, b1, a1 - b1, x0, b0, a0 - b0])

total = {c: [sum(r[i] for r in rows if r[1] == c) for i in (2, 4, 7)] for c in ('base', 'ext1')}
assert len(rows) == 2 * 99 and total['base'][0] == 23729 and sum(total['base'][1:]) == 23728
assert total['ext1'][0] == 39434 and sum(total['ext1'][1:]) == 39434

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'replay_counts.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['slot', 'component', 'attempted', 'x1', 'b1', 'unclassified1', 'x0', 'b0', 'unclassified0'])
    w.writerows(rows)
design = {'datasets': len(datasets), 'n': N, 'm': M, 'replay_records': len(replays)}
(OUT / 'design.json').write_text(json.dumps(design) + '\n', encoding='utf-8')
