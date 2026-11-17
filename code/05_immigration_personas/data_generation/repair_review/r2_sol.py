#!/usr/bin/env python3
"""Second step of the aligned pipeline with gpt-6.1-sol on a ChatGPT Pro login (the authors' choice of 4 November),
in place of Miao et al.'s GPT-5.4: a summary of each repaired analysis, then three independent PASS/FAIL reviews of it
(prompts in r2_prompts.py, unchanged).

Sessions follow the project's Sol read runner (religion-journal-20261030/channel/claimread/code/claimread_gl.py):
Codex CLI 0.165.0, a fresh private folder per session, auth copied from sol-card-main-gl/auth/<account>, and a
bubblewrap that hides /gpfs, /home, /nfs, /scratch and /tmp. Review sessions have no tools. Summary sessions have a
shell, the repaired script as analysis.py, its output as results.csv, the data read-only at the path the script reads,
and the project's curated Python; nothing about persona, model or the repair. New claims stop when the account's
primary meter reaches 97%.

Usage: python r2_sol.py --cohort s46_miao --phase summary|review --account primary|second [--workers 6]
       python r2_sol.py --probe primary
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
sys.path.insert(0, str(HERE))
from r2_prompts import review_prompt, summary_prompt, verdict  # noqa: E402

G = Path('<cluster scratch>')
CODEX = G / 'codex-cli-0.165.0/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex'
PYDIR, PYLIB = '<cluster python 3.12.1>/bin', '<cluster python 3.12.1>/lib'
CURATED = G / 'immigration-scientific-python-v1'
LOCAL = Path('/tmp') / f"miao-sol-{os.environ.get('SLURM_JOB_ID', 'login')}-{os.getpid()}"
PROXY, NO_PROXY = 'http://<campus proxy>:3128', 'localhost,127.0.0.1,::1'
MODEL, EFFORT, PAUSE_AT, ATTEMPTS = 'gpt-6.1-sol', 'high', 97.0, 3
WALL = {'summary': 1800, 'review': 900}
BASE_DISABLED = ('apps', 'auth_elicitation', 'browser_use', 'browser_use_external', 'browser_use_full_cdp_access',
                 'computer_use', 'fast_mode', 'goals', 'guardian_approval', 'hooks', 'image_generation', 'in_app_browser',
                 'in_app_chat', 'in_app_dictation', 'in_app_updates', 'memories', 'multi_agent', 'multi_agent_v2',
                 'plugin_sharing', 'plugins', 'remote_plugin', 'shell_snapshot', 'skill_mcp_dependency_install',
                 'skill_search', 'tool_call_mcp_elicitation', 'tool_suggest', 'unbounded_connection_retries',
                 'view_image', 'workspace_dependencies')
NO_TOOLS = ('shell_tool', 'unified_exec', 'code_mode_host')
BINARIES = ('codex', 'codex-code-mode-host')


def install():
    LOCAL.mkdir(parents=True, exist_ok=True)
    for b in BINARIES:
        shutil.copy2(CODEX.parent / b, LOCAL / b)
LOCK = threading.Lock()
STOP = threading.Event()


def config(tools):
    v = {'model': MODEL, 'model_reasoning_effort': EFFORT, 'approval_policy': 'never',
         'sandbox_mode': 'danger-full-access' if tools else 'read-only', 'web_search': 'disabled',
         'project_doc_max_bytes': 0, 'project_doc_fallback_filenames': [],
         'memories.generate_memories': False, 'memories.use_memories': False}
    v.update({f'features.{n}': False for n in BASE_DISABLED + (() if tools else NO_TOOLS)})
    if tools:  # as in multi-model-originals-20261030/code/run_codex.py
        v.update({f'features.{n}': True for n in NO_TOOLS})
    return ''.join(f'{k} = {json.dumps(x)}\n' for k, x in sorted(v.items()))


def session(account, prompt, *, tools=False, files=None, ro=(), wall=900, keep=None):
    """One fresh session; returns stdout, stderr, return code, meter, and the text of `keep` in the workspace."""
    root = LOCAL / f'ws-{uuid.uuid4().hex[:16]}'
    for name in ('workspace', 'home', 'tmp', 'codex-home'):
        (root / name).mkdir(parents=True)
    for b in BINARIES:  # the shell runs through codex-code-mode-host, which Codex expects beside itself
        os.link(LOCAL / b, root / b)
    (root / 'codex-home/config.toml').write_text(config(tools))
    auth = json.loads((G / 'sol-card-main-gl/auth' / account / 'auth.json').read_text())
    if auth.get('auth_mode') != 'chatgpt':
        raise RuntimeError('ChatGPT subscription login required')
    auth['OPENAI_API_KEY'] = None
    (root / 'codex-home/auth.json').write_text(json.dumps(auth))
    os.chmod(root / 'codex-home/auth.json', 0o600)
    for name, src in (files or {}).items():
        shutil.copy2(src, root / 'workspace' / name)
    command = [str(root / 'codex'), 'exec', '--cd', str(root / 'workspace'), '--strict-config', '--ignore-rules',
               '--model', MODEL, '--skip-git-repo-check', '--json']
    hidden = [x for p in ('/gpfs', '/home', '/nfs', '/scratch', '/tmp') if Path(p).is_dir() and not Path(p).is_symlink()
              for x in ('--tmpfs', p)]
    shown = [x for p in ro for x in ('--ro-bind', str(p), str(p))]
    wrap = ['/usr/bin/bwrap', '--dev-bind', '/', '/', *hidden, '--bind', str(root), str(root), *shown,
            '--die-with-parent', '--']
    env = {'PATH': f'{PYDIR}:/usr/bin:/bin', 'HOME': str(root / 'home'), 'CODEX_HOME': str(root / 'codex-home'),
           'TMPDIR': str(root / 'tmp'), 'LANG': 'en_US.UTF-8', 'NO_PROXY': NO_PROXY, 'no_proxy': NO_PROXY,
           **{k: PROXY for k in ('HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy')}}
    if tools:
        env.update(PYTHONPATH=str(CURATED), LD_LIBRARY_PATH=PYLIB, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2')
    proc = subprocess.Popen(wrap + command + ['-'], cwd=root / 'workspace', env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(prompt, timeout=wall)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        stdout, stderr = proc.communicate()
    meter = None
    for f in (root / 'codex-home/sessions').rglob('*.jsonl') if (root / 'codex-home/sessions').exists() else ():
        for line in f.read_text().splitlines():
            i = line.find('"rate_limits":')
            if i >= 0:
                snap, _ = json.JSONDecoder().raw_decode(line, i + len('"rate_limits":'))
                meter = snap or meter
    kept = (root / 'workspace' / keep).read_text() if keep and (root / 'workspace' / keep).exists() else None
    code = proc.returncode
    shutil.rmtree(root, ignore_errors=True)
    return stdout, stderr, code, meter, kept


def events(stdout):
    ev = [json.loads(x) for x in stdout.splitlines() if x.strip().startswith('{')]
    life = [e['type'] for e in ev if e.get('type') in ('turn.started', 'turn.completed', 'turn.failed')]
    answers = [e['item']['text'] for e in ev if e.get('type') == 'item.completed'
               and e.get('item', {}).get('type') == 'agent_message']
    usage = [e.get('usage') for e in ev if e.get('type') == 'turn.completed']
    sid = next((e.get('thread_id') for e in ev if e.get('type') == 'thread.started'), None)
    tools = sum(1 for e in ev if e.get('type') == 'item.completed'
                and e.get('item', {}).get('type') in ('command_execution', 'function_call', 'local_shell_call'))
    return (bool(life) and life[-1] == 'turn.completed'), (answers[-1] if answers else ''), \
        (usage[-1] if usage else None), sid, tools


def run_task(t, a, data, data_kind, log):
    run, kind, k = t
    for attempt in range(1, ATTEMPTS + 1):
        if STOP.is_set() or (TOP / 'STOP').exists():
            return
        started = time.time()
        if kind == 'summary':
            prompt = summary_prompt('analysis.py', 'results.csv', str(data), 'summary.md')
            out = session(a.account, prompt, tools=True, wall=WALL[kind], keep='summary.md', ro=(data, CURATED),
                          files={'analysis.py': run / 'analysis_final_revised.py',
                                 'results.csv': run / 'results_final_revised.csv'})
        else:
            prompt = review_prompt((run / 'r2_summary.md').read_text(), data_kind)
            out = session(a.account, prompt, wall=WALL[kind])
        stdout, stderr, code, meter, kept = out
        done, answer, usage, sid, tools = events(stdout)
        if kind == 'summary':
            ok = done and kept is not None and len(kept) > 200
            if ok:
                (run / 'r2_summary.md').write_text(kept.strip() + '\n')
        else:
            ok = done and verdict(answer) is not None and tools == 0
            if ok:
                (run / f'r2_verdict_{k}.txt').write_text(answer.strip() + '\n')
        used = ((meter or {}).get('primary') or {}).get('used_percent')
        if used is not None and used >= PAUSE_AT:
            STOP.set()
        with LOCK, open(log, 'a') as fh:
            fh.write(json.dumps({'run': f'{run.parent.name}/{run.name}', 'task': kind, 'k': k, 'attempt': attempt,
                                 'ok': bool(ok), 'account': a.account, 'model': MODEL, 'effort': EFFORT,
                                 'session_id': sid, 'tool_items': tools, 'usage': usage, 'meter_primary': used,
                                 'returncode': code, 'seconds': round(time.time() - started, 1),
                                 'stderr_tail': None if ok else stderr[-600:]}) + '\n')
        if ok:
            return


def probe(account):
    install()
    stdout, stderr, code, meter, _ = session(account, 'Reply with the single word ready.', wall=300)
    done, answer, usage, sid, tools = events(stdout)
    print(json.dumps({'account': account, 'completed': done, 'answer': answer[:40], 'returncode': code,
                      'primary_used_percent': ((meter or {}).get('primary') or {}).get('used_percent'),
                      'primary_resets_at': ((meter or {}).get('primary') or {}).get('resets_at'),
                      'credits': (meter or {}).get('credits')}))
    shutil.rmtree(LOCAL, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort')
    ap.add_argument('--phase', choices=['summary', 'review'])
    ap.add_argument('--account', choices=['primary', 'second'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--probe', choices=['primary', 'second'])
    ap.add_argument('--limit', type=int, default=0, help='first N tasks only (live tests)')
    a = ap.parse_args()
    if a.probe:
        return probe(a.probe)
    base = TOP / 'base' / a.cohort
    data = next((base / 'data').glob('*.csv')).resolve()
    data_kind = 'miao' if data.name == 'data_clean.csv' else 'clean'
    runs = sorted(p for p in (base / 'results').glob('*/run_*')
                  if (p / 'analysis_final_revised.py').exists() and (p / 'results_final_revised.csv').exists())
    if a.phase == 'summary':
        tasks = [(r, 'summary', None) for r in runs if not (r / 'r2_summary.md').exists()]
    else:
        tasks = [(r, 'review', k) for r in runs if (r / 'r2_summary.md').exists() for k in (1, 2, 3)
                 if not (r / f'r2_verdict_{k}.txt').exists()]
    log = TOP / 'results' / f'r2_sol_{a.cohort}_{a.phase}.jsonl'
    log.parent.mkdir(exist_ok=True)
    print(f'{a.cohort} {a.phase}: {len(runs)} repaired runs, {len(tasks)} tasks, {data.name} ({data_kind}), '
          f'account {a.account}', flush=True)
    tasks = tasks[:a.limit] if a.limit else tasks
    install()
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda t: run_task(t, a, data, data_kind, log), tasks))
    finally:
        shutil.rmtree(LOCAL, ignore_errors=True)
    print('stopped at the meter' if STOP.is_set() else 'done', flush=True)


if __name__ == '__main__':
    main()
