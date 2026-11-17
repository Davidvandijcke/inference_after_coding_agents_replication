#!/usr/bin/env python3
"""Methodological-soundness review of Card replay analyses with gpt-6.1-sol on the authors'
ChatGPT Codex subscription (same login handling as the Terra runner). One review per
replay: a fresh Codex home, no tools, the prompt in inputs/review_prompt.txt, a
structured {pass, reason} answer. Append-only and resumable; stops claiming new
reviews at PAUSE_AT percent of the weekly meter or when runs/<name>/STOP exists.

  caffeinate -is python3 code/review.py --name pilot --cases 20 --workers 4
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'card-terra-subscription-pilot-20261029/code'))
import pilot  # noqa: E402  (the Terra runner: disabled features, event parsing, CLI path)

MODEL, REASONING = 'gpt-6.1-sol', 'high'
PAUSE_AT = 80.0
WALL_SECONDS = 600
SCHEMA = {'type': 'object', 'properties': {'pass': {'type': 'boolean'}, 'reason': {'type': 'string'}},
          'required': ['pass', 'reason'], 'additionalProperties': False}
INSTRUCTIONS = 'Answer from the material in the prompt. Do not run commands or use tools.'
PROMPT = (HERE / 'inputs/review_prompt.txt').read_text()
DESIGN = (HERE.parent / 'card-terra-subscription-pilot-20261029/inputs/DESIGN.md').read_text().strip()
CODEBOOK = (HERE.parent / 'card-terra-subscription-pilot-20261029/inputs/CODEBOOK.md').read_text().strip()
lock = threading.Lock()
state = {'paused': False, 'used_percent': None}


def config_text(root):
    values = {'model': MODEL, 'model_reasoning_effort': REASONING, 'approval_policy': 'never',
              'sandbox_mode': 'read-only', 'web_search': 'disabled', 'project_doc_max_bytes': 0,
              'project_doc_fallback_filenames': [], 'memories.generate_memories': False,
              'memories.use_memories': False, 'developer_instructions': INSTRUCTIONS,
              'shell_environment_policy.inherit': 'none'}
    values.update({f'features.{name}': False for name in pilot.DISABLED_FEATURES})
    values.update({'features.code_mode_host': False, 'features.shell_tool': False, 'features.unified_exec': False})
    return '\n'.join(f'{k} = {json.dumps(v)}' for k, v in sorted(values.items())) + '\n'


def new_root():
    root = Path('/private/tmp') / f'rv-{uuid.uuid4().hex[:16]}'
    for name in ('workspace', 'home', 'tmp', 'codex-home'):
        (root / name).mkdir(parents=True)
    (root / 'codex-home/config.toml').write_text(config_text(root))
    auth = json.loads((Path.home() / '.codex/auth.json').read_text())
    if auth.get('auth_mode') != 'chatgpt':
        raise RuntimeError('ChatGPT subscription login required')
    auth['OPENAI_API_KEY'] = None
    path = root / 'codex-home/auth.json'
    path.write_text(json.dumps(auth)); os.chmod(path, 0o600)
    return root


def prompt_for(packet):
    code = packet['code'] or '(The implementation was not recovered; judge from the declaration.)'
    return (PROMPT.replace('{{DESIGN}}', DESIGN).replace('{{CODEBOOK}}', CODEBOOK)
            .replace('{{DECLARATION}}', packet['declaration']).replace('{{CODE}}', code))


def review(packet, out):
    root = new_root()
    rec = {'replay_id': packet['replay_id'], 'model': MODEL, 'route': 'chatgpt_subscription', 'started': time.time()}
    try:
        prompt = prompt_for(packet)
        (root / 'schema.json').write_text(json.dumps(SCHEMA))
        cmd = [pilot.CODEX, 'exec', '--cd', str(root / 'workspace'), '--strict-config', '--ignore-rules', '--model', MODEL,
               '--skip-git-repo-check', '--json', '--output-schema', str(root / 'schema.json'), '-']
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(root / 'home'), 'CODEX_HOME': str(root / 'codex-home'),
               'TMPDIR': str(root / 'tmp'), 'LANG': 'en_US.UTF-8'}
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, cwd=root / 'workspace',
                              timeout=WALL_SECONDS)
        case_dir = out / 'cases' / packet['replay_id']; case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'events.jsonl').write_text(proc.stdout)
        (case_dir / 'stderr.txt').write_text(proc.stderr[-20000:])
        if proc.returncode != 0:
            raise RuntimeError(f'returncode {proc.returncode}')
        parsed = pilot.parse_events(proc.stdout)
        answer = json.loads(parsed['answer'])
        if set(answer) != {'pass', 'reason'} or type(answer['pass']) is not bool:
            raise ValueError('Malformed answer')
        tools = sum(1 for line in proc.stdout.splitlines() if '"command_execution"' in line or '"file_change"' in line)
        rec.update(status='complete', verdict=answer, usage=parsed['usage'], tool_events=tools,
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    except Exception as error:  # recorded, never retried automatically
        rec.update(status='unavailable', error=str(error)[-800:])
    finally:
        auth = root / 'codex-home/auth.json'
        if auth.exists():
            auth.unlink()
    snap = pilot_rate_limits(root)
    rec['rate_limits'] = snap
    if snap:
        used = (snap.get('primary') or {}).get('used_percent')
        state['used_percent'] = used
        if (used is not None and used >= PAUSE_AT) or snap.get('rate_limit_reached_type') or snap.get('spend_control_reached'):
            state['paused'] = True
    shutil.rmtree(root, ignore_errors=True)
    rec['seconds'] = round(time.time() - rec['started'], 1)
    with lock:
        with open(out / 'results.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
    return rec


def pilot_rate_limits(root):
    latest = None
    for f in (Path(root) / 'codex-home/sessions').rglob('*.jsonl'):
        for line in f.read_text().splitlines():
            i = line.find('"rate_limits":')
            if i >= 0:
                snap, _ = json.JSONDecoder().raw_decode(line, i + len('"rate_limits":'))
                latest = snap or latest
    return latest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True)
    ap.add_argument('--cases', type=int); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2026102910)
    ap.add_argument('--ids', help='file with replay IDs to review (e.g. the stability repeat)')
    ap.add_argument('--retry-unavailable', action='store_true', help='review again any replay whose review failed')
    args = ap.parse_args()
    out = HERE / 'runs' / args.name; out.mkdir(parents=True, exist_ok=True)
    packets = [json.loads(l) for l in open(HERE / 'inputs/packets.jsonl')]
    random.Random(args.seed).shuffle(packets)
    done = set()
    if (out / 'results.jsonl').exists():
        done = {r['replay_id'] for r in map(json.loads, open(out / 'results.jsonl'))
                if r['status'] == 'complete' or not args.retry_unavailable}
    if args.ids:
        keep = set(Path(args.ids).read_text().split())
        packets = [p for p in packets if p['replay_id'] in keep]
    todo = [p for p in packets if p['replay_id'] not in done][:args.cases]
    queue = iter(todo)

    def worker():
        while not state['paused'] and not (out / 'STOP').exists():
            with lock:
                p = next(queue, None)
            if p is None:
                return
            review(p, out)

    with ThreadPoolExecutor(args.workers) as ex:
        for _ in range(args.workers):
            ex.submit(worker)
    rows = [json.loads(l) for l in open(out / 'results.jsonl')]
    latest = {r['replay_id']: r for r in rows}  # a retried review supersedes its failed attempt
    print(json.dumps({'replays': len(latest), 'complete': sum(r['status'] == 'complete' for r in latest.values()),
                      'weekly_used_percent': state['used_percent'], 'paused': state['paused']}))


if __name__ == '__main__':
    main()
