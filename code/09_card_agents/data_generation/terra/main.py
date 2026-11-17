#!/usr/bin/env python3
"""Fixed 3,000-replay Terra cohort (MAIN_PLAN.md) on the authors' ChatGPT Codex
subscriptions, with the pilot's per-case protocol (code/pilot.py).

Resumable and append-only: a case with a recorded start is never rerun, except
one whose analysis turn was refused for a usage or rate limit before the agent
acted. Each ChatGPT account runs its own workers. Before each new case a
worker checks its account's weekly usage meter from that account's latest
finished session and stops claiming at PAUSE_AT percent, so ChatGPT credits are
never drawn; the other accounts carry on. Touch runs/main/STOP to stop claiming
by hand. Each finished case is packed into runs/main/cases/<case_id>.tar.gz and
its result appended to runs/main/results.jsonl.

  caffeinate -is python3 code/main.py --account primary=~/.codex:16 --account second=~/<second account config>:16
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot  # noqa: E402

MAIN = pilot.PILOT / 'runs/main'
STARTED, RESULTS, CASES, STOP = MAIN / 'started.jsonl', MAIN / 'results.jsonl', MAIN / 'cases', MAIN / 'STOP'
PAUSE_AT = 85.0
USAGE_TEXT = ('usage_limit', 'usage limit', 'hit your usage limit', 'rate_limit_reached')
RATE_TEXT = ('status":429', 'status\\":429', 'rate_limit', 'Too Many Requests')
RETRY_ELIGIBLE = ('unstarted_usage_limit', 'unstarted_rate_limited')
lock = threading.Lock()
state = {}


def append(path, record):
    with lock, path.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def rate_limits(root):
    """Latest rate-limit snapshot written in this run's session files."""
    latest = None
    for f in (Path(root) / 'codex-home/sessions').rglob('*.jsonl'):
        for line in f.read_text().splitlines():
            i = line.find('"rate_limits":')
            if i >= 0:
                snap, _ = json.JSONDecoder().raw_decode(line, i + len('"rate_limits":'))
                if snap:
                    latest = snap
    return latest


def session_models(root):
    models, efforts = set(), set()
    for f in (Path(root) / 'codex-home/sessions').rglob('*.jsonl'):
        for line in f.read_text().splitlines():
            x = json.loads(line)
            if x.get('type') == 'turn_context':
                models.add(x['payload'].get('model')); efforts.add(x['payload'].get('effort'))
    return sorted(m for m in models if m), sorted(e for e in efforts if e)


def mentions(phase_dir, needles):
    text = ''.join((phase_dir / n).read_text() for n in ('events.jsonl', 'stderr.txt') if (phase_dir / n).exists())
    return any(s in text for s in needles)


def agent_acted(phase_dir):
    path = phase_dir / 'events.jsonl'
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        if line.startswith('{'):
            x = json.loads(line)
            if x.get('type') == 'item.completed' and (x.get('item') or {}).get('type') != 'error':
                return True
    return False


def pack(case_id, root, record):
    """Archive the phase records, session files and workspace artifacts (the
    packet's data.csv is rebuilt from the roster), then remove the run folders."""
    path = CASES / f'{case_id}.tar.gz'
    if path.exists():  # an earlier attempt refused before the agent acted
        path.rename(path.with_name(f'{case_id}.refused-{uuid.uuid4().hex[:8]}.tar.gz'))
    with tarfile.open(path, 'x:gz') as tar:
        tar.add(record, arcname='record')
        tar.add(Path(root) / 'codex-home/sessions', arcname='sessions')
        tar.add(Path(root) / 'workspace', arcname='workspace', filter=lambda t: None if t.name == 'workspace/data.csv' else t)
    with tarfile.open(path) as tar:
        tar.getmembers()
    shutil.rmtree(root)
    shutil.rmtree(record)
    return str(path.relative_to(pilot.PILOT))


def run_case(case, target, account):
    name, home = account
    route = f'chatgpt_subscription:{name}'
    append(STARTED, {'case_id': case['case_id'], 'position': case['position'], 'route': route, 'at': time.time()})
    if pilot.sha(case['packet']) != case['packet_sha256']:
        raise RuntimeError('Packet changed')
    root = pilot.new_case(Path(case['packet']), auth_file=home / 'auth.json', isolate=True)
    record = pilot.RUN_PARENT / f'rec-{uuid.uuid4().hex[:16]}'
    data_sha = pilot.sha(root / 'workspace/data.csv')
    result = {'case_id': case['case_id'], 'position': case['position'], 'route': route, 'model': pilot.MODEL,
              'cli': pilot.CODEX, 'read_isolation': True}
    try:
        analysis, committed, commitment = pilot.analysis_and_commitment(root, record)
        prompt = (pilot.INPUTS / 'comparison_prompt.txt').read_text().replace(
            '{{TARGET_JSON}}', json.dumps(target, ensure_ascii=False, separators=(',', ':')))
        compared = pilot.invoke(root, prompt, record / 'comparison', session_id=analysis['session_id'],
                                schema=pilot.COMPARISON_SCHEMA)
        verdict = json.loads(compared['answer'])
        if set(verdict) != {'model_null_match', 'reason'} or type(verdict['model_null_match']) is not bool:
            raise ValueError('Malformed comparison')
        result.update(status='complete', session_id=analysis['session_id'], commitment=commitment, verdict=verdict,
                      usage={'analysis': analysis['usage'], 'commitment': committed['usage'],
                             'comparison': compared['usage']})
    except BaseException as error:
        phases = [record / p for p in ('analysis', 'commitment', 'comparison')]
        unstarted = not agent_acted(phases[0]) and not phases[1].exists()
        if any(mentions(p, USAGE_TEXT) for p in phases):
            result['status'] = 'unstarted_usage_limit' if unstarted else 'interrupted_usage_limit'
            state[name]['paused'] = True
        elif any(mentions(p, RATE_TEXT) for p in phases):
            result['status'] = 'unstarted_rate_limited' if unstarted else 'interrupted_rate_limited'
        else:
            result['status'] = 'unavailable'
        result['error'] = str(error)[-1500:]
    finally:
        pilot.remove_auth(root)
    result['data_unchanged'] = pilot.sha(root / 'workspace/data.csv') == data_sha
    result['session_models'], result['session_efforts'] = session_models(root)
    snap = rate_limits(root)
    result['rate_limits'] = snap
    if snap:
        used = (snap.get('primary') or {}).get('used_percent')
        state[name]['used_percent'] = used
        if (used is not None and used >= PAUSE_AT) or snap.get('rate_limit_reached_type') or snap.get('spend_control_reached'):
            state[name]['paused'] = True
    result['archive'] = pack(case['case_id'], root, record)
    append(RESULTS, result)
    return result


def main(accounts):
    MAIN.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(exist_ok=True)
    roster = json.loads((pilot.PILOT / 'main-roster.json').read_text())
    original = json.loads((pilot.RUNS / 'original/result.json').read_text())
    if original.get('status') != 'committed':
        raise SystemExit('Committed original required')
    for name, home, _ in accounts:
        auth = json.loads((home / 'auth.json').read_text())
        if auth.get('auth_mode') != 'chatgpt':
            raise SystemExit(f'{name}: ChatGPT login required')
        state[name] = {'paused': False, 'used_percent': None}
    results = {r['case_id']: r for r in read_jsonl(RESULTS)}
    started = {r['case_id'] for r in read_jsonl(STARTED)}
    for case_id in started - set(results):
        # Started in an earlier run that ended before a result: no retry.
        record = {'case_id': case_id, 'status': 'unavailable', 'error': 'runner ended before the case finished'}
        append(RESULTS, record)
        results[case_id] = record
    queue = deque(c for c in roster['cases']
                  if c['case_id'] not in results or results[c['case_id']]['status'] in RETRY_ELIGIBLE)

    def worker(name, home):
        while True:
            if STOP.exists() or state[name]['paused']:
                return
            with lock:
                case = queue.popleft() if queue else None
            if case is None:
                return
            try:
                result = run_case(case, original['commitment'], (name, home))
            except Exception as error:
                # Runner-side failure: the case keeps its start and gets no retry.
                append(RESULTS, {'case_id': case['case_id'], 'position': case['position'], 'status': 'unavailable',
                                 'error': f'runner: {error}'[-1500:]})
                continue
            if result['status'] in RETRY_ELIGIBLE:
                with lock:
                    queue.append(case)
                time.sleep(60)

    jobs = [(name, home) for name, home, n in accounts for _ in range(n)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for f in [pool.submit(worker, *job) for job in jobs]:
            f.result()
    final = {}
    for r in read_jsonl(RESULTS):
        final[r['case_id']] = r['status']
    counts = {s: list(final.values()).count(s) for s in sorted(set(final.values()))}
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'accounts': state, 'stop_file': STOP.exists(),
                      'remaining': sum(1 for c in roster['cases'] if final.get(c['case_id']) in (None,) + RETRY_ELIGIBLE),
                      'statuses': counts}))


def account(text):
    name, rest = text.split('=', 1)
    home, workers = rest.rsplit(':', 1)
    return name, Path(home).expanduser(), int(workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', type=account, action='append',
                        help='name=CODEX_HOME:workers, repeatable; default primary=~/.codex:8')
    args = parser.parse_args()
    main(args.account or [('primary', Path.home() / '.codex', 8)])
