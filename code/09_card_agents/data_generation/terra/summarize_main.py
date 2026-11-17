#!/usr/bin/env python3
"""Main-cohort summary (MAIN_PLAN.md): merges the Mac results (runs/main/) and
the Great Lakes results (runs/main/gl/out/), writes one row per roster case to
processed/terra-replays.csv, and lists every agent command that reaches outside
its own run root in processed/outside-commands.jsonl. No model calls.
Run from the run root: python3 code/summarize_main.py"""
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
import tarfile

RUN = Path(__file__).resolve().parents[1]
MAIN, GL = RUN / 'runs/main', RUN / 'runs/main/gl'
PROCESSED = RUN / 'processed'
PHASES = ('analysis', 'commitment', 'comparison')
# Project, home, storage and other runs' paths, upward escapes, and root or /tmp listings.
OUTSIDE = re.compile(r'/Users/|Dropbox|inference_after|card-|terra-|/private/tmp|/tmp\b|/gpfs/|/scratch|'
                     r'/home/|/nfs/|(?<![\w.])\.\.(?=/|\s|$|\'|")|\bfind\s+/(?=\s)|\bls\s+(-\S+\s+)*/(?=\s|$)')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def archive_members(spec):
    """Yields (member name, bytes) for one case archive, Mac or Great Lakes."""
    if ':' in spec:
        outer, inner = spec.split(':', 1)
        with tarfile.open(GL / outer) as tar:
            data = tar.extractfile(inner).read()
        fileobj = io.BytesIO(data)
    else:
        fileobj = open(RUN / spec, 'rb')
    with fileobj, tarfile.open(fileobj=fileobj, mode='r:gz') as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.startswith('record/'):
                yield m.name, tar.extractfile(m).read()


def outside_commands(spec):
    """Commands reaching outside the run root, and the number of commands the
    Great Lakes sandbox refused (a race in Codex's deny-glob scan)."""
    files = dict(archive_members(spec))
    hits, refused = [], 0
    for phase in PHASES:
        phase_json = files.get(f'record/{phase}/phase.json')
        if phase_json is None:
            continue
        argv = json.loads(phase_json)['argv']
        workspace = argv[argv.index('--cd') + 1]
        root = workspace.rsplit('/workspace', 1)[0]
        for line in files.get(f'record/{phase}/events.jsonl', b'').decode().splitlines():
            if not line.startswith('{'):
                continue
            x = json.loads(line)
            item = x.get('item') or {}
            if x.get('type') != 'item.completed' or item.get('type') != 'command_execution':
                continue
            refused += 'error building bubblewrap command' in (item.get('aggregated_output') or '')
            command = item.get('command', '').replace(root, '<root>')
            if OUTSIDE.search(command):
                hits.append({'phase': phase, 'command': command, 'exit_code': item.get('exit_code'),
                             'output': (item.get('aggregated_output') or '').replace(root, '<root>')[:1500]})
    return hits, refused


def tokens(usage):
    total = Counter()
    for phase in (usage or {}).values():
        total.update(phase)
    return total


roster = json.loads((RUN / 'main-roster.json').read_text())
records = read_jsonl(MAIN / 'results.jsonl') + [r for p in sorted(GL.glob('out/**/results-*.jsonl'), key=str)
                                                for r in read_jsonl(p)]
started = read_jsonl(MAIN / 'started.jsonl') + [r for p in sorted(GL.glob('out/**/started-*.jsonl'), key=str)
                                                for r in read_jsonl(p)]
by_case = {}
for r in records:
    by_case.setdefault(r['case_id'], []).append(r)
starts = Counter(s['case_id'] for s in started)
mac_ids = {r['case_id'] for r in read_jsonl(MAIN / 'results.jsonl')}
gl_ids = set(by_case) - mac_ids
assert not mac_ids & {r['case_id'] for p in GL.glob('out/**/results-*.jsonl') for r in read_jsonl(p)}, 'case on both routes'

PROCESSED.mkdir(exist_ok=True)
rows, audit = [], []
for case in roster['cases']:
    history = by_case.get(case['case_id'], [])
    final = history[-1] if history else {'status': 'not_run'}
    row = {'case_id': case['case_id'], 'position': case['position'], 'bank_id': case['bank_id'],
           'stratum': case['stratum'], 'bank_p': case['bank_p'], 'status': final['status'],
           'machine': 'mac' if case['case_id'] in mac_ids else 'great_lakes' if history else '',
           'route': final.get('route', ''), 'read_isolation': bool(final.get('read_isolation')),
           'records': len(history), 'starts': starts[case['case_id']],
           'data_unchanged': final.get('data_unchanged'),
           'models': '|'.join(final.get('session_models') or []), 'efforts': '|'.join(final.get('session_efforts') or [])}
    if final['status'] == 'complete':
        t = tokens(final['usage'])
        row.update(match=final['verdict']['model_null_match'], reason=final['verdict']['reason'],
                   **{f'commit_{k}': final['commitment'].get(k, '') for k in
                      ('model', 'null', 'population', 'outcome', 'sample_restrictions')},
                   input_tokens=t['input_tokens'], cached_input_tokens=t['cached_input_tokens'],
                   output_tokens=t['output_tokens'], reasoning_output_tokens=t['reasoning_output_tokens'])
    else:
        row['error'] = (final.get('error') or '')[:300]
    if 'archive' in final:
        hits, row['sandbox_refused'] = outside_commands(final['archive'])
        row['outside_commands'] = len(hits)
        audit += [{'case_id': case['case_id'], 'read_isolation': row['read_isolation'], 'machine': row['machine'], **h}
                  for h in hits]
    rows.append(row)

fields = list(dict.fromkeys(k for r in rows for k in r))
with open(PROCESSED / 'terra-replays.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
(PROCESSED / 'outside-commands.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in audit))

summary = {'u': roster['u'], 'planned': Counter(r['stratum'] for r in rows),
           'status': Counter(r['status'] for r in rows), 'machine': Counter(r['machine'] for r in rows),
           'route': Counter(r['route'] for r in rows), 'read_isolation': Counter(r['read_isolation'] for r in rows),
           'multiple_records': sum(r['records'] > 1 for r in rows), 'multiple_starts': sum(r['starts'] > 1 for r in rows),
           'data_changed': sum(r['data_unchanged'] is False for r in rows),
           'models': Counter(r['models'] for r in rows), 'efforts': Counter(r['efforts'] for r in rows)}
for s in ('tail', 'body'):
    done = [r for r in rows if r['stratum'] == s and r['status'] == 'complete']
    summary[s] = {'attempted': sum(r['stratum'] == s for r in rows), 'classified': len(done),
                  'excluded': sum(r['stratum'] == s and r['status'] != 'complete' for r in rows),
                  'matches': sum(r['match'] for r in done)}
done = [r for r in rows if r['status'] == 'complete']
summary['mean_tokens_per_replay'] = {k: round(sum(r[k] for r in done) / len(done)) for k in
                                     ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens')}
summary['sandbox_refused'] = {'cases': sum(bool(r.get('sandbox_refused')) for r in rows),
                              'commands': sum(r.get('sandbox_refused', 0) for r in rows)}
summary['cases_with_outside_commands'] = Counter(f"{r['machine']}, read_isolation={r['read_isolation']}"
                                                 for r in rows if r.get('outside_commands'))
(RUN / 'results').mkdir(exist_ok=True)
(RUN / 'results/main-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
