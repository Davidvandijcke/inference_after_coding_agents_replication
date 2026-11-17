#!/usr/bin/env python3
"""Audit of the Card replay review with Claude Opus 5.5 at high effort on the authors' Max
subscription, through the Claude Code CLI with the long-lived token from `claude setup-token`
in the repository's git-ignored .env, as in the Miao replication's subscription route. One call
per replay: an empty home and working folder, a minimal environment, no tools, no settings or
memory, the prompt in inputs/audit_prompt.txt. The pilot takes 40 random
reviewer passes and 10 failures (5 the hand audit confirmed, 5 it did not). Append-only and
resumable; stops at a usage-limit error or when runs/<name>/STOP exists. The full run is in gl/.

  caffeinate -is python3 code/audit.py --name pilot --cases 50 --workers 3
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parents[1]
OUT = HERE.parent
MODEL, EFFORT = 'claude-opus-5-5', 'high'
SEED = 2026103001
WALL_SECONDS = 600
SYSTEM = 'You audit empirical analyses. Answer from the material in the prompt. Do not use tools.'
PROMPT = (HERE / 'inputs/audit_prompt_pilot.txt').read_text()
DESIGN = (OUT / 'card-terra-subscription-pilot-20261029/inputs/DESIGN.md').read_text().strip()
CODEBOOK = (OUT / 'card-terra-subscription-pilot-20261029/inputs/CODEBOOK.md').read_text().strip()
PACKETS = {json.loads(l)['replay_id']: json.loads(l) for l in open(OUT / 'card-soundness-review-20261029/inputs/packets.jsonl')}
REVIEW = {r['replay_id']: r for r in csv.DictReader(open(OUT / 'card-soundness-review-20261029/results/card-replay-review.csv'))}
TOKEN = None
lock = threading.Lock()
stopped = threading.Event()


def pilot_cases():
    rng = random.Random(SEED)
    ids = sorted(k for k in REVIEW if PACKETS[k]['code'])
    confirmed = [k for k in ids if REVIEW[k]['audit'] == 'confirmed']
    other = [k for k in ids if REVIEW[k]['audit'] in ('disclaimer', 'no_effect', 'choice')]
    passes = [k for k in ids if REVIEW[k]['pass'] == 'true']
    cases = rng.sample(confirmed, 5) + rng.sample(other, 5) + rng.sample(passes, 40)
    rng.shuffle(cases)
    return cases


def prompt_for(rid):
    p, v = PACKETS[rid], REVIEW[rid]
    return (PROMPT.replace('{{DESIGN}}', DESIGN).replace('{{CODEBOOK}}', CODEBOOK)
            .replace('{{DECLARATION}}', p['declaration']).replace('{{CODE}}', p['code'])
            .replace('{{VERDICT}}', 'pass' if v['pass'] == 'true' else 'fail').replace('{{REASON}}', v['reason']))


def parse(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(text)


def token():
    line = next(l for l in (OUT.parent / '.env').read_text().splitlines() if l.startswith('CLAUDE_SETUP_TOKEN='))
    return line.split('=', 1)[1].strip().strip('"\'')


def audit(rid):
    cmd = ['claude', '-p', '--model', MODEL, '--effort', EFFORT, '--output-format', 'json', '--tools', '',
           '--system-prompt', SYSTEM, '--setting-sources', '', '--strict-mcp-config', '--disable-slash-commands',
           '--no-session-persistence']
    started = time.time()
    with tempfile.TemporaryDirectory(prefix='audit-', dir='/private/tmp') as cwd:
        env = {'HOME': cwd, 'TMPDIR': cwd, 'PATH': os.environ['PATH'], 'LANG': 'en_US.UTF-8', 'CLAUDE_CODE_OAUTH_TOKEN': TOKEN,
               'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1'}
        run = subprocess.run(cmd, input=prompt_for(rid), capture_output=True, text=True, cwd=cwd, env=env, timeout=WALL_SECONDS)
    row = {'replay_id': rid, 'model': MODEL, 'effort': EFFORT, 'seconds': round(time.time() - started, 1), 'status': 'unavailable'}
    try:
        r = json.loads(run.stdout)
        row.update({k: r.get(k) for k in ('is_error', 'num_turns', 'total_cost_usd', 'usage', 'modelUsage')})
        row['text'] = r.get('result')
        if not r.get('is_error'):
            row['verdict'] = parse(r['result']); row['status'] = 'complete'
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        row['error'] = f'{type(e).__name__}: {run.stdout[-300:]} {run.stderr[-300:]}'
    return row


def worker(rid, path, stop_file):
    if stopped.is_set() or stop_file.exists():
        return
    row = audit(rid)
    with lock:
        with open(path, 'a') as f:
            f.write(json.dumps(row) + '\n')
    if row['status'] != 'complete' and 'limit' in json.dumps(row).lower():
        stopped.set()


def main():
    global TOKEN
    TOKEN = token()
    a = argparse.ArgumentParser(); a.add_argument('--name'); a.add_argument('--cases', type=int); a.add_argument('--workers', type=int, default=3)
    args = a.parse_args()
    run = HERE / 'runs' / args.name; run.mkdir(parents=True, exist_ok=True)
    path = run / 'results.jsonl'
    done = {json.loads(l)['replay_id'] for l in open(path)} if path.exists() else set()
    cases = [c for c in pilot_cases()[:args.cases] if c not in done]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda c: worker(c, path, run / 'STOP'), cases))
    print(json.dumps({'run': args.name, 'attempted': len(cases), 'stopped': stopped.is_set()}))


if __name__ == '__main__':
    main()
