#!/usr/bin/env python3
"""One array task of the Opus audit on Great Lakes. Takes every TASKS-th case from cases.jsonl,
skips replays already complete in results/, and runs one no-tool Claude Code call per replay in a
fresh node-local home, on the authors' Max subscription. Appends to results/task-<id>.jsonl, one file
per task. Stops claiming new cases at a usage-limit error or when STOP exists."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

RUN = Path(os.environ['RUN'])
BIN, LOCAL = os.environ['CLAUDE_BIN'], os.environ['LOCAL_TMP']
TASK, TASKS = int(os.environ['SLURM_ARRAY_TASK_ID']), int(os.environ['TASKS'])
MODEL, EFFORT = 'claude-opus-5-5', 'high'
SYSTEM = 'You audit empirical analyses. Answer from the material in the prompt. Do not use tools.'
WALL_SECONDS = 600
TOKEN = (RUN / 'private/claude-token').read_text().strip()
PROXY = {k: v for k, v in os.environ.items() if k.lower() in ('http_proxy', 'https_proxy', 'no_proxy')}
lock = threading.Lock()
stopped = threading.Event()


def parse(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(text)


def call(case):
    cmd = [BIN, '-p', '--model', MODEL, '--effort', EFFORT, '--output-format', 'json', '--tools', '', '--system-prompt', SYSTEM,
           '--setting-sources', '', '--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence']
    started = time.time()
    row = {'replay_id': case['replay_id'], 'model': MODEL, 'effort': EFFORT, 'task': TASK, 'status': 'unavailable'}
    with tempfile.TemporaryDirectory(prefix='a-', dir=LOCAL) as cwd:
        env = {'HOME': cwd, 'TMPDIR': cwd, 'PATH': '/usr/bin:/bin', 'LANG': 'en_US.UTF-8', 'CLAUDE_CODE_OAUTH_TOKEN': TOKEN,
               'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1', 'DISABLE_AUTOUPDATER': '1', **PROXY}
        try:
            run = subprocess.run(cmd, input=case['prompt'], capture_output=True, text=True, cwd=cwd, env=env, timeout=WALL_SECONDS)
            r = json.loads(run.stdout)
            row.update({k: r.get(k) for k in ('is_error', 'num_turns', 'total_cost_usd', 'usage')})
            row['models'] = sorted(r.get('modelUsage') or {}); row['text'] = r.get('result')
            if not r.get('is_error'):
                row['verdict'] = parse(r['result']); row['status'] = 'complete'
        except (json.JSONDecodeError, TypeError, KeyError, subprocess.TimeoutExpired) as e:
            row['error'] = f'{type(e).__name__}'
    row['seconds'] = round(time.time() - started, 1)
    return row


def work(case, path):
    for attempt in range(3):
        if stopped.is_set() or (RUN / 'STOP').exists():
            return
        row = call(case); row['attempt'] = attempt
        with lock:
            with open(path, 'a') as f:
                f.write(json.dumps(row) + '\n')
        if row['status'] == 'complete':
            return
        if 'limit' in str(row.get('text')).lower():
            stopped.set(); return
        time.sleep(20 * (attempt + 1))


def main():
    a = argparse.ArgumentParser(); a.add_argument('--workers', type=int, default=6); a.add_argument('--limit', type=int)
    args = a.parse_args()
    results = RUN / 'results'; results.mkdir(exist_ok=True)
    done = set()
    for p in results.glob('task-*.jsonl'):
        done |= {r['replay_id'] for r in map(json.loads, open(p)) if r['status'] == 'complete'}
    cases = [json.loads(l) for i, l in enumerate(open(RUN / 'cases.jsonl')) if i % TASKS == TASK]
    cases = [c for c in cases if c['replay_id'] not in done][:args.limit]
    time.sleep(TASK * 5)  # staggered start
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda c: work(c, results / f'task-{TASK}.jsonl'), cases))
    print(json.dumps({'task': TASK, 'cases': len(cases), 'stopped': stopped.is_set()}))


if __name__ == '__main__':
    main()
