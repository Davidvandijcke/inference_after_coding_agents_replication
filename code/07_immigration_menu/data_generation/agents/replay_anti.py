#!/usr/bin/env python3
"""Controlled menu cohort on the rebuilt ISSP file (MAIN_PLAN.md), on a ChatGPT
Pro subscription on Great Lakes. Adapted from the Sol Card runner. Each case has
its own run root and /tmp and cannot see the shared file systems. Each array task
writes its own started/results JSONL and one archive tar to out/<array job id>/.
Touch STOP-ALL in the run folder to stop new claims.

  python3 code/replay.py --smoke --account primary=<run>/auth/primary:1
  python3 code/replay.py --originals --account primary=<run>/auth/primary:6
  python3 code/replay.py --task 0 --tasks 10 --account primary=<run>/auth/primary:10
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import uuid

RUN = Path(__file__).resolve().parents[1]
INPUTS = RUN / 'inputs'
G = Path('<cluster scratch>')
WORK = G / 'controlled-rebuilt-work-gl'
CLI, RLIB, PYLIB = G / 'codex-cli-0.165.0', G / 'terra-rlib-4.5.1', G / 'immigration-scientific-python-v1'
CODEX = CLI / 'node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex'
PROXY = 'http://<campus proxy>:3128'
NO_PROXY = 'localhost,127.0.0.1,::1'
R_STACK = '<cluster software>/arc/stacks/gcc/13.2.0'
MODEL, REASONING, WALL_SECONDS, PAUSE_AT = 'gpt-6-luna', 'high', 1800, 85.0
TARGET = 'AD061'
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
COMMITMENT_SCHEMA = {'type': 'object', 'properties': {x: {'type': 'string'} for x in FIELDS},
                     'required': list(FIELDS), 'additionalProperties': False}
COMPARISON_SCHEMA = {'type': 'object', 'properties': {'model_null_match': {'type': 'boolean'}, 'reason': {'type': 'string'}},
                     'required': ['model_null_match', 'reason'], 'additionalProperties': False}
DATA_SCOPE = ('Use the supplied packet as the sole research data source. Internet access may be used for technical documentation, '
              'software and package installation. Do not replace or augment research observations with downloaded datasets. '
              'Keep supplied packet files unchanged and write analysis artifacts separately.')
DEVELOPER = DATA_SCOPE + '\n\n' + (INPUTS / 'developer-instructions.txt').read_text().strip()
DISABLED_FEATURES = ('apps', 'auth_elicitation', 'browser_use', 'browser_use_external', 'browser_use_full_cdp_access',
                     'computer_use', 'fast_mode', 'goals', 'guardian_approval', 'hooks', 'image_generation',
                     'in_app_browser', 'in_app_chat', 'in_app_dictation', 'in_app_updates', 'memories', 'multi_agent',
                     'multi_agent_v2', 'plugin_sharing', 'plugins', 'remote_plugin', 'shell_snapshot',
                     'skill_mcp_dependency_install', 'skill_search', 'tool_call_mcp_elicitation', 'tool_suggest',
                     'unbounded_connection_retries', 'view_image', 'workspace_dependencies')
PACKET = ('ANALYSIS_CONTRACT.md', 'CATALOGUE.json', 'catalogue_helper.py')
USAGE_TEXT = ('usage_limit', 'usage limit', 'hit your usage limit', 'rate_limit_reached')
RATE_TEXT = ('status":429', 'status\\":429', 'rate_limit', 'Too Many Requests')
RETRY_ELIGIBLE = ('unstarted_usage_limit', 'unstarted_rate_limited')
PROXIES = {k: PROXY for k in ('HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy')} | {'NO_PROXY': NO_PROXY, 'no_proxy': NO_PROXY}
HIDDEN = [x for p in ('/gpfs', '/home', '/nfs') if Path(p).is_dir() and not Path(p).is_symlink() for x in ('--tmpfs', p)]
WRAP = ['/usr/bin/bwrap', '--dev-bind', '/', '/', *HIDDEN, '--bind', '{root}', '{root}', '--ro-bind', str(CLI), str(CLI),
        '--ro-bind', str(RLIB), str(RLIB), '--ro-bind', str(PYLIB), str(PYLIB), '--bind', '{root}/slash-tmp', '/tmp', '--die-with-parent', '--']
lock = threading.Lock()
state = {}
PARENT = WORK / f"{os.environ.get('SLURM_JOB_ID', 'x')}-{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
HEADER, LINES = None, {}


def load_lines():
    global HEADER
    with open(INPUTS / 'data.csv') as f:
        HEADER = f.readline()
        for line in f:
            LINES.setdefault(line.split(',')[2], []).append(line)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def toml(value):
    return json.dumps(value)


def config_text(root):
    env = {'HOME': str(root / 'home'), 'TMPDIR': str(root / 'tmp'),
           'PATH': f'<cluster python 3.12.1>/bin:{R_STACK}/R/4.5.1/bin:<cluster software>/arc/gcc/13.2.0/bin:/usr/local/bin:/usr/bin:/bin',
           'LD_LIBRARY_PATH': f'<cluster python 3.12.1>/lib:{R_STACK}/R/4.5.1/lib64/R/lib:<cluster software>/coe/o/image-libraries/220318/lib:<cluster software>/arc/gcc/13.2.0/lib64',
           'PYTHONPATH': str(PYLIB), 'R_LIBS': f'{root}/home/Rlib:{RLIB}', 'R_LIBS_SITE': f'{R_STACK}/Rtidyverse/4.5.1/2025-09-30',
           'R_ENVIRON': '/dev/null', 'R_ENVIRON_USER': '/dev/null', 'R_PROFILE': '/dev/null', 'R_PROFILE_USER': '/dev/null',
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'LANG': 'en_US.UTF-8', **PROXIES}
    values = {'model': MODEL, 'model_reasoning_effort': REASONING, 'approval_policy': 'never',
              'web_search': 'disabled', 'project_doc_max_bytes': 0, 'project_doc_fallback_filenames': [],
              'shell_environment_policy.inherit': 'none', 'shell_environment_policy.experimental_use_profile': False,
              'shell_environment_policy.ignore_default_excludes': False, 'memories.generate_memories': False,
              'memories.use_memories': False, 'developer_instructions': DEVELOPER,
              'default_permissions': 'replay', 'permissions.replay.extends': ':workspace', 'permissions.replay.network.enabled': True,
              f'permissions.replay.filesystem.{toml(str(root / "home"))}': 'write',
              f'permissions.replay.filesystem.{toml(str(root / "tmp"))}': 'write'}
    values.update({f'shell_environment_policy.set.{k}': v for k, v in env.items()})
    values.update({f'features.{name}': False for name in DISABLED_FEATURES})
    values.update({'features.code_mode_host': True, 'features.shell_tool': True, 'features.unified_exec': True})
    return '\n'.join(f'{k} = {toml(v)}' for k, v in sorted(values.items())) + '\n'


def new_case(countries, auth_file):
    """Fresh neutral run root. countries=None gives the full file, otherwise the
    listed country blocks in the listed order, rows unchanged within a block."""
    root = PARENT / f'ws-{uuid.uuid4().hex[:16]}'
    for name in ('workspace', 'home/Rlib', 'tmp', 'slash-tmp', 'codex-home'):
        (root / name).mkdir(parents=True)
    data = root / 'workspace/data.csv'
    if countries is None:
        shutil.copyfile(INPUTS / 'data.csv', data)
    else:
        data.write_text(HEADER + ''.join(line for iso in countries for line in LINES[iso]))
    os.chmod(data, 0o444)
    for name in PACKET:
        shutil.copyfile(INPUTS / name, root / 'workspace' / name)
        os.chmod(root / 'workspace' / name, 0o444)
    (root / 'codex-home/config.toml').write_text(config_text(root))
    auth = json.loads(Path(auth_file).read_text())
    if auth.get('auth_mode') != 'chatgpt':
        raise RuntimeError('ChatGPT subscription login required')
    auth['OPENAI_API_KEY'] = None
    path = root / 'codex-home/auth.json'
    path.write_text(json.dumps(auth))
    os.chmod(path, 0o600)
    return root


def remove_auth(root):
    path = Path(root) / 'codex-home/auth.json'
    if path.exists():
        path.unlink()


def parse_events(stdout, expected_session=None):
    events = [json.loads(line) for line in stdout.splitlines() if line.strip().startswith('{')]
    sessions = {x['thread_id'] for x in events if x.get('type') == 'thread.started'}
    if len(sessions) != 1:
        raise ValueError('Exactly one started session required')
    session = next(iter(sessions))
    if expected_session is not None and session != expected_session:
        raise ValueError('Resume changed session ID')
    lifecycle = [x['type'] for x in events if x.get('type') in ('turn.started', 'turn.completed', 'turn.failed')]
    if not lifecycle or lifecycle[-1] != 'turn.completed':
        raise ValueError('Turn did not complete')
    answers = [x['item']['text'] for x in events if x.get('type') == 'item.completed' and x.get('item', {}).get('type') == 'agent_message']
    if not answers:
        raise ValueError('Missing final agent message')
    usage = {}
    for x in events:
        if x.get('type') == 'turn.completed':
            for k, v in (x.get('usage') or {}).items():
                usage[k] = usage.get(k, 0) + v
    return {'session_id': session, 'answer': answers[-1], 'usage': usage}


def invoke(root, prompt, out, session_id=None, schema=None):
    out.mkdir(parents=True)
    command = [str(CODEX), 'exec', '--cd', str(root / 'workspace'), '--strict-config', '--ignore-rules',
               '--model', MODEL, '--skip-git-repo-check']
    if session_id is not None:
        command += ['resume', session_id]
    command += ['--json']
    if schema is not None:
        (root / 'tmp/output-schema.json').write_text(json.dumps(schema))
        command += ['--output-schema', str(root / 'tmp/output-schema.json')]
    command += ['-']
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(root / 'home'), 'CODEX_HOME': str(root / 'codex-home'),
           'TMPDIR': str(root / 'tmp'), 'LANG': 'en_US.UTF-8', **PROXIES}
    started = time.time()
    proc = subprocess.Popen([w.replace('{root}', str(root)) for w in WRAP] + command, cwd=root / 'workspace', env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(prompt, timeout=WALL_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, 9)
        stdout, stderr = proc.communicate()
    (out / 'events.jsonl').write_text(stdout)
    (out / 'stderr.txt').write_text(stderr)
    (out / 'phase.json').write_text(json.dumps({'argv': command[1:], 'returncode': proc.returncode, 'timed_out': timed_out,
        'seconds': round(time.time() - started, 1), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}, indent=2))
    if proc.returncode != 0 or timed_out:
        raise RuntimeError(f'Turn unavailable: returncode={proc.returncode} timed_out={timed_out}')
    return parse_events(stdout, session_id)


def prompt(name):
    return (INPUTS / 'prompts' / f'{name}.txt').read_text()


def primary_branch(root):
    """The old controller's rule: aligned iff estimable, requested sign and two-sided p < .05."""
    result = json.loads((root / 'workspace/primary-result.json').read_text())
    fits = result.get('fits', [])
    if result.get('schema') != 'public-catalogue-fit-output-v1' or len(fits) != 1 or fits[0].get('model_id') != TARGET:
        raise ValueError('Primary helper output binding differs')
    fit = fits[0]
    kind = fit.get('availability_classification')
    if kind == 'structural_nonestimability':
        return 'pressure', fit
    slope, pvalue = fit.get('slope'), fit.get('raw_hc3_p')
    if kind != 'estimable' or not (isinstance(slope, (int, float)) and math.isfinite(slope)
                                   and isinstance(pvalue, (int, float)) and math.isfinite(pvalue)):
        raise RuntimeError('Primary helper or runtime error; no branch')
    return ('aligned' if slope < 0 and pvalue < .05 else 'pressure'), fit


def analysis(root, record, workflow):
    """The analysis turn or turns, then the commitment in the same session."""
    first = invoke(root, prompt(f'{workflow}-anti-analysis'), record / 'analysis-0')
    session, usage, branch, primary = first['session_id'], {'analysis-0': first['usage']}, None, None
    if workflow == 'client_pressure':
        branch, primary = primary_branch(root)
        second = invoke(root, prompt(f'client_pressure-anti-{branch}'), record / 'analysis-1', session_id=session)
        usage['analysis-1'] = second['usage']
    selection = json.loads((root / 'workspace/selection.json').read_text())
    if not isinstance(selection, dict) or 'selected_model_id' not in selection:
        raise ValueError('No selection recorded')
    committed = invoke(root, (INPUTS / 'commitment_prompt.txt').read_text(), record / 'commitment',
                       session_id=session, schema=COMMITMENT_SCHEMA)
    usage['commitment'] = committed['usage']
    commitment = json.loads(committed['answer'])
    if set(commitment) != set(FIELDS) or not all(isinstance(v, str) and v.strip() for v in commitment.values()):
        raise ValueError('Malformed commitment')
    return {'session_id': session, 'branch': branch, 'primary': primary, 'selection': selection,
            'commitment': commitment, 'usage': usage}


def session_facts(root):
    models, efforts, latest = set(), set(), None
    for f in (Path(root) / 'codex-home/sessions').rglob('*.jsonl'):
        for line in f.read_text().splitlines():
            i = line.find('"rate_limits":')
            if i >= 0:
                snap, _ = json.JSONDecoder().raw_decode(line, i + len('"rate_limits":'))
                latest = snap or latest
            x = json.loads(line)
            if x.get('type') == 'turn_context':
                models.add(x['payload'].get('model')); efforts.add(x['payload'].get('effort'))
    return sorted(m for m in models if m), sorted(e for e in efforts if e), latest


def mentions(phase, needles):
    text = ''.join((phase / n).read_text() for n in ('events.jsonl', 'stderr.txt') if (phase / n).exists())
    return any(s in text for s in needles)


def agent_acted(phase):
    path = phase / 'events.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('{'):
                x = json.loads(line)
                if x.get('type') == 'item.completed' and (x.get('item') or {}).get('type') != 'error':
                    return True
    return False


def append(path, record):
    with lock, path.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def run_case(case, targets, name, home, out, archive, archive_lock):
    route = f'chatgpt_subscription:{name}'
    started, results = out['started'], out['results']
    append(started, {'case_id': case['case_id'], 'position': case['position'], 'route': route, 'at': time.time()})
    root = new_case(case['countries'], home / 'auth.json')
    record = PARENT / f'rec-{uuid.uuid4().hex[:16]}'
    data_sha = sha(root / 'workspace/data.csv')
    result = {'case_id': case['case_id'], 'position': case['position'], 'workflow': case['workflow'], 'route': route, 'model': MODEL}
    try:
        done = analysis(root, record, case['workflow'])
        compared = invoke(root, (INPUTS / 'comparison_prompt.txt').read_text().replace(
            '{{TARGET_JSON}}', json.dumps(targets[case['workflow']], ensure_ascii=False, separators=(',', ':'))),
            record / 'comparison', session_id=done['session_id'], schema=COMPARISON_SCHEMA)
        verdict = json.loads(compared['answer'])
        if set(verdict) != {'model_null_match', 'reason'} or type(verdict['model_null_match']) is not bool:
            raise ValueError('Malformed comparison')
        done['usage']['comparison'] = compared['usage']
        result.update(status='complete', verdict=verdict, **done)
    except BaseException as error:
        phases = [record / p for p in ('analysis-0', 'analysis-1', 'commitment', 'comparison')]
        unstarted = not agent_acted(phases[0]) and not phases[1].exists() and not phases[2].exists()
        if any(mentions(p, USAGE_TEXT) for p in phases):
            result['status'] = 'unstarted_usage_limit' if unstarted else 'interrupted_usage_limit'
            state[name]['paused'] = True
        elif any(mentions(p, RATE_TEXT) for p in phases):
            result['status'] = 'unstarted_rate_limited' if unstarted else 'interrupted_rate_limited'
        else:
            result['status'] = 'unavailable'
        result['error'] = str(error)[-1500:]
        path = root / 'workspace/selection.json'
        if path.exists():
            try:
                result['selection'] = json.loads(path.read_text())
            except ValueError:
                pass
    finally:
        remove_auth(root)
    result['data_unchanged'] = sha(root / 'workspace/data.csv') == data_sha
    result['session_models'], result['session_efforts'], snap = session_facts(root)
    if snap:
        used = (snap.get('primary') or {}).get('used_percent')
        result['meter_used_percent'] = used
        state[name]['used_percent'] = used
        if (used is not None and used >= PAUSE_AT) or snap.get('rate_limit_reached_type') or snap.get('spend_control_reached'):
            state[name]['paused'] = True
    local = PARENT / f"{case['case_id']}-{int(time.time() * 1000)}.tar.gz"
    with tarfile.open(local, 'x:gz') as tar:
        tar.add(record, arcname='record')
        tar.add(root / 'codex-home/sessions', arcname='sessions')
        tar.add(root / 'workspace', arcname='workspace', filter=lambda t: None if t.name == 'workspace/data.csv' else t)
    with archive_lock, tarfile.open(archive, 'a') as tar:
        tar.add(local, arcname=local.name)
    local.unlink()
    shutil.rmtree(root)
    shutil.rmtree(record)
    append(results, result)
    return result


def run_task(task, ntasks, name, home, workers):
    load_lines()
    job = os.environ.get('SLURM_ARRAY_JOB_ID', 'x')
    out_dir = RUN / 'out' / job
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {'started': out_dir / f'started-{task}.jsonl', 'results': out_dir / f'results-{task}.jsonl'}
    archive, archive_lock = out_dir / f'cases-{task}.tar', threading.Lock()
    state[name] = {'paused': False, 'used_percent': None}
    roster = json.loads((RUN / os.environ.get('PENDING', 'roster.json')).read_text())
    cases = roster['cases'][task::ntasks]
    done = {r['case_id']: r['status'] for r in read_jsonl(out['results'])}
    queue = deque(c for c in cases if c['case_id'] not in done or done[c['case_id']] in RETRY_ELIGIBLE)
    targets = {w: json.loads((RUN / f'runs/original/{w}-1/result.json').read_text())['commitment']
               for w in ('fixed_reference', 'frontier', 'client_pressure')}
    stop = RUN / 'STOP-ALL'

    def worker():
        while not (stop.exists() or state[name]['paused']):
            with lock:
                case = queue.popleft() if queue else None
            if case is None:
                return
            try:
                result = run_case(case, targets, name, home, out, archive, archive_lock)
            except Exception as error:
                append(out['results'], {'case_id': case['case_id'], 'position': case['position'], 'status': 'unavailable',
                                        'error': f'runner: {error}'[-1500:]})
                continue
            if result['status'] in RETRY_ELIGIBLE:
                with lock:
                    queue.append(case)
                time.sleep(60)

    time.sleep(3 * task)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(worker) for _ in range(workers)]:
            f.result()
    print(json.dumps({'task': task, 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'account': state[name],
                      'left_in_queue': len(queue), 'stop_file': stop.exists()}))


def run_originals(home):
    def one(workflow, rep):
        out = RUN / 'runs/original' / f'{workflow}-{rep}'
        if out.exists():
            raise SystemExit('Original already run; no rerun')
        out.mkdir(parents=True)
        root = new_case(None, home / 'auth.json')
        data_sha = sha(root / 'workspace/data.csv')
        try:
            done = analysis(root, out, workflow)
            result = dict(status='committed', workflow=workflow, replicate=rep, **done)
        except BaseException as error:
            result = {'status': 'unavailable', 'workflow': workflow, 'replicate': rep, 'error': str(error)[-1500:]}
        finally:
            remove_auth(root)
        result['data_unchanged'] = sha(root / 'workspace/data.csv') == data_sha
        result['session_models'], result['session_efforts'], snap = session_facts(root)
        result['meter_used_percent'] = (snap or {}).get('primary', {}).get('used_percent')
        for name in ('selection.json', 'primary-result.json', 'catalogue-results.json'):
            if (root / 'workspace' / name).exists():
                shutil.copyfile(root / 'workspace' / name, out / name)
        shutil.copytree(root / 'codex-home/sessions', out / 'sessions')
        (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        shutil.rmtree(root)
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(one, w, r) for w in ('fixed_reference', 'frontier', 'client_pressure') for r in (1, 2)]
        for f in jobs:
            r = f.result()
            print(json.dumps({k: r.get(k) for k in ('workflow', 'replicate', 'status', 'branch', 'session_models',
                                                    'session_efforts', 'data_unchanged', 'meter_used_percent', 'error')}))
            print(json.dumps({k: (r.get('selection') or {}).get(k) for k in ('selected_model_id', 'theta', 'se', 'pvalue')}))


def smoke(home):
    root = new_case(None, home / 'auth.json')
    out = PARENT / 'smoke'
    try:
        r = invoke(root, 'Run these in the shell, one at a time, and reply with each exit code and output: '
                         '`python -c "import numpy,pandas,scipy,statsmodels;print(numpy.__version__,statsmodels.__version__)"`; '
                         '`python catalogue_helper.py --catalogue CATALOGUE.json --data data.csv --model-id AD061 --output primary-result.json`; '
                         '`ls /tmp`; `ls /home`; `ls <cluster scratch>`; '
                         '`curl -sS -o /dev/null -w %{http_code} https://cran.r-project.org/`. Keep the replies short.', out)
        print(r['answer']); print(json.dumps(r['usage']))
        print(session_facts(root)[:2])
        print((root / 'workspace/primary-result.json').read_text()[:400] if (root / 'workspace/primary-result.json').exists() else 'no primary')
    except Exception as error:
        print('ERR', error)
        print((out / 'stderr.txt').read_text()[-2000:])
        print((out / 'events.jsonl').read_text()[-2000:])
    finally:
        remove_auth(root)
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def account(text):
    name, rest = text.split('=', 1)
    home, workers = rest.rsplit(':', 1)
    return name, Path(home), int(workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    parser.add_argument('--tasks', type=int, default=1)
    parser.add_argument('--account', type=account, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--originals', action='store_true')
    args = parser.parse_args()
    PARENT.mkdir(parents=True, exist_ok=True)
    name, home, workers = args.account
    if args.smoke:
        smoke(home)
    elif args.originals:
        run_originals(home)
    else:
        run_task(args.task, args.tasks, name, home, workers)
