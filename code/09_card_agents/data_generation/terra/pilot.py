#!/usr/bin/env python3
"""Local Card selective-replay pilot on gpt-6.1-terra through the authors' Codex
(ChatGPT) subscription. No API key is used: each run gets a fresh Codex home
whose auth copy has its API key removed.

The protocol follows the 3,000-replay luna cohort: the same research,
commitment and comparison prompts, the same packet files, the same Codex
settings where this CLI accepts them, an analysis turn, a commitment turn
resumed in the same session, and for replays a comparison turn resumed again.
Differences from the Great Lakes runs: the model, a local macOS Codex sandbox
(workspace-write with network, instead of bubblewrap), Codex CLI 0.146.0, and a
local R library. Run directories are neutral (/private/tmp/ws-<hex>) and
contain no role, stratum or target labels.

  python3 code/pilot.py original
  python3 code/pilot.py replays --workers 6
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import time
import uuid

PILOT = Path(__file__).resolve().parents[1]
INPUTS = PILOT / 'inputs'
RUNS = PILOT / 'runs'
# The CLI bundled with the ChatGPT app; older CLIs are refused gpt-6.1-terra.
# On Great Lakes the REPLAY_* environment variables (set by code/gl_task.sbatch)
# replace the Mac defaults below.
CODEX = os.environ.get('REPLAY_CODEX', '<codex-cli>')
RUN_PARENT = Path(os.environ.get('REPLAY_RUN_PARENT', '/private/tmp'))
MODEL = 'gpt-6.1-terra'
REASONING = 'high'
WALL_SECONDS = 1800
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
COMMITMENT_SCHEMA = {'type': 'object', 'properties': {x: {'type': 'string'} for x in FIELDS},
                     'required': list(FIELDS), 'additionalProperties': False}
COMPARISON_SCHEMA = {'type': 'object', 'properties': {'model_null_match': {'type': 'boolean'},
    'reason': {'type': 'string'}}, 'required': ['model_null_match', 'reason'], 'additionalProperties': False}
DATA_SCOPE = ('Use the supplied packet as the sole research data source. Internet access may be used for technical documentation, '
              'software and package installation. Do not replace or augment research observations with downloaded datasets. '
              'Keep supplied packet files unchanged and write analysis artifacts separately.')
# The luna cohort's disabled features that this CLI still knows (personality
# and remote_compaction_v2 are removed in 0.164).
DISABLED_FEATURES = ('apps', 'auth_elicitation', 'browser_use', 'browser_use_external', 'browser_use_full_cdp_access',
                     'computer_use', 'fast_mode', 'goals', 'guardian_approval', 'hooks', 'image_generation',
                     'in_app_browser', 'in_app_chat', 'in_app_dictation', 'in_app_updates', 'memories', 'multi_agent',
                     'multi_agent_v2', 'plugin_sharing', 'plugins', 'remote_plugin', 'shell_snapshot',
                     'skill_mcp_dependency_install', 'skill_search', 'tool_call_mcp_elicitation', 'tool_suggest',
                     'unbounded_connection_retries', 'view_image', 'workspace_dependencies')
R_USER_LIB = str(Path.home() / 'Library/R/arm64/4.5/library')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def toml(value):
    return json.dumps(value)


HOME = str(Path.home())
# Paths an isolated run may not read: project folders, other agents' scratch
# areas, Codex and Claude homes, the packet store, and every run's Codex home
# and records (which hold login copies and other runs' transcripts).
DENY_READ = json.loads(os.environ['REPLAY_DENY_READ']) if 'REPLAY_DENY_READ' in os.environ else [
    f'{HOME}/<synced folder 1>', f'{HOME}/<synced folder 2>', f'{HOME}/Documents',
    f'{HOME}/Desktop', f'{HOME}/Downloads', f'{HOME}/.codex', f'{HOME}/<second account config>', f'{HOME}/.claude',
    f'{HOME}/<assistant workspace>', '<packet-cache>/terra-card-main', '<agent-scratch>']
DENY_READ += [f'{RUN_PARENT}/ws-*/codex-home', f'{RUN_PARENT}/rec-*']
# Agent shell variables to add or replace; '{root}' is the run root.
AGENT_ENV = json.loads(os.environ.get('REPLAY_AGENT_ENV', '{}'))
# Extra variables for the Codex process itself (proxy settings on Great Lakes).
CODEX_ENV = json.loads(os.environ.get('REPLAY_CODEX_ENV', '{}'))
# Command prefix for the Codex process (on Great Lakes, an outer bubblewrap
# that gives it an ordinary /tmp in place of Slurm's remapped one).
WRAP = json.loads(os.environ.get('REPLAY_WRAP', '[]'))


def config_text(root, isolate=False):
    env = {'HOME': str(root / 'home'), 'TMPDIR': str(root / 'tmp'),
           'PATH': '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
           'R_LIBS': f'{root / "home/Rlib"}:{R_USER_LIB}',
           'R_ENVIRON': '/dev/null', 'R_ENVIRON_USER': '/dev/null', 'R_PROFILE': '/dev/null',
           'R_PROFILE_USER': '/dev/null', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'LANG': 'en_US.UTF-8'}
    env.update({k: v.replace('{root}', str(root)) for k, v in AGENT_ENV.items()})
    values = {'model': MODEL, 'model_reasoning_effort': REASONING,
              'approval_policy': 'never', 'sandbox_mode': 'workspace-write',
              'sandbox_workspace_write.network_access': True,
              'sandbox_workspace_write.writable_roots': [str(root / 'home'), str(root / 'tmp')],
              'web_search': 'disabled', 'project_doc_max_bytes': 0, 'project_doc_fallback_filenames': [],
              'shell_environment_policy.inherit': 'none',
              'shell_environment_policy.experimental_use_profile': False,
              'shell_environment_policy.ignore_default_excludes': False,
              'memories.generate_memories': False, 'memories.use_memories': False,
              'developer_instructions': DATA_SCOPE}
    values.update({f'shell_environment_policy.set.{k}': v for k, v in env.items()})
    values.update({f'features.{name}': False for name in DISABLED_FEATURES})
    values.update({'features.code_mode_host': True, 'features.shell_tool': True, 'features.unified_exec': True})
    if isolate:
        # Same writable areas and network as workspace-write, with read denials.
        for key in ('sandbox_mode', 'sandbox_workspace_write.network_access', 'sandbox_workspace_write.writable_roots'):
            del values[key]
        values.update({'default_permissions': 'replay', 'permissions.replay.extends': ':workspace',
                       'permissions.replay.network.enabled': True})
        for path, rule in [(str(root / 'home'), 'write'), (str(root / 'tmp'), 'write')] + [(p, 'deny') for p in DENY_READ]:
            values[f'permissions.replay.filesystem.{toml(path)}'] = rule
    return '\n'.join(f'{k} = {toml(v)}' for k, v in sorted(values.items())) + '\n'


def new_case(data_csv, auth_file=None, isolate=False):
    """Fresh neutral run root with the packet, an empty home and a fresh Codex home.
    auth_file selects which ChatGPT login to copy (default ~/.codex/auth.json);
    isolate adds the read denials in DENY_READ."""
    root = RUN_PARENT / f'ws-{uuid.uuid4().hex[:16]}'
    for name in ('workspace', 'home/Rlib', 'tmp', 'codex-home'):
        (root / name).mkdir(parents=True)
    for name, source in (('data.csv', data_csv), ('CODEBOOK.md', INPUTS / 'CODEBOOK.md'),
                         ('DESIGN.md', INPUTS / 'DESIGN.md')):
        shutil.copyfile(source, root / 'workspace' / name)
        os.chmod(root / 'workspace' / name, 0o444)
    (root / 'codex-home/config.toml').write_text(config_text(root, isolate))
    auth = json.loads(Path(auth_file or Path.home() / '.codex/auth.json').read_text())
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
    answers = [x['item']['text'] for x in events if x.get('type') == 'item.completed'
               and x.get('item', {}).get('type') == 'agent_message']
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
    command = [CODEX, 'exec', '--cd', str(root / 'workspace'), '--strict-config', '--ignore-rules',
               '--model', MODEL, '--skip-git-repo-check']
    if session_id is not None:
        command += ['resume', session_id]
    command += ['--json']
    if schema is not None:
        (out / 'schema.json').write_text(json.dumps(schema))
        command += ['--output-schema', str(out / 'schema.json')]
    command += ['-']
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(root / 'home'),
           'CODEX_HOME': str(root / 'codex-home'), 'TMPDIR': str(root / 'tmp'), 'LANG': 'en_US.UTF-8', **CODEX_ENV}
    started = time.time()
    proc = subprocess.Popen(WRAP + command, cwd=root / 'workspace', env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(prompt, timeout=WALL_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, 9)
        stdout, stderr = proc.communicate()
    (out / 'events.jsonl').write_text(stdout)
    (out / 'stderr.txt').write_text(stderr)
    receipt = {'argv': command[1:], 'returncode': proc.returncode, 'timed_out': timed_out,
               'seconds': round(time.time() - started, 1), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    (out / 'phase.json').write_text(json.dumps(receipt, indent=2))
    if proc.returncode != 0 or timed_out:
        raise RuntimeError(f'Turn unavailable: returncode={proc.returncode} timed_out={timed_out}')
    for name in ('data.csv', 'CODEBOOK.md', 'DESIGN.md'):
        if sha(root / 'workspace' / name) != sha(INPUTS / name) and name != 'data.csv':
            raise RuntimeError('Packet file changed')
    return parse_events(stdout, session_id)


def analysis_and_commitment(root, out):
    prompt = (INPUTS / 'research-prompt.txt').read_text()
    analysis = invoke(root, prompt, out / 'analysis')
    committed = invoke(root, (INPUTS / 'commitment_prompt.txt').read_text(), out / 'commitment',
                       session_id=analysis['session_id'], schema=COMMITMENT_SCHEMA)
    commitment = json.loads(committed['answer'])
    if set(commitment) != set(FIELDS) or not all(isinstance(v, str) and v.strip() for v in commitment.values()):
        raise ValueError('Malformed commitment')
    return analysis, committed, commitment


def run_original():
    out = RUNS / 'original'
    if out.exists():
        raise SystemExit('Original already run; no rerun')
    out.mkdir(parents=True)
    root = new_case(INPUTS / 'data.csv')
    data_sha = sha(root / 'workspace/data.csv')
    try:
        analysis, committed, commitment = analysis_and_commitment(root, out)
        program = root / 'workspace/analysis.R'
        if not program.is_file():
            raise ValueError('No analysis.R emitted')
        shutil.copyfile(program, out / 'emitted-analysis.R')
        result = {'status': 'committed', 'model': MODEL, 'reasoning_effort': REASONING,
                  'session_id': analysis['session_id'], 'commitment': commitment,
                  'usage': {'analysis': analysis['usage'], 'commitment': committed['usage']},
                  'data_unchanged': sha(root / 'workspace/data.csv') == data_sha,
                  'program_sha256': sha(out / 'emitted-analysis.R'), 'run_root': str(root)}
    except BaseException as error:
        result = {'status': 'unavailable', 'error': str(error), 'run_root': str(root)}
    finally:
        remove_auth(root)
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_replay(case, target):
    out = RUNS / 'replays' / case['case_id']
    if out.exists():
        return {'case_id': case['case_id'], 'status': 'previously_started_no_retry'}
    out.mkdir(parents=True)
    root = new_case(PILOT / case['packet'])
    data_sha = sha(root / 'workspace/data.csv')
    try:
        analysis, committed, commitment = analysis_and_commitment(root, out)
        prompt = (INPUTS / 'comparison_prompt.txt').read_text().replace(
            '{{TARGET_JSON}}', json.dumps(target, ensure_ascii=False, separators=(',', ':')))
        compared = invoke(root, prompt, out / 'comparison', session_id=analysis['session_id'],
                          schema=COMPARISON_SCHEMA)
        verdict = json.loads(compared['answer'])
        if set(verdict) != {'model_null_match', 'reason'} or type(verdict['model_null_match']) is not bool:
            raise ValueError('Malformed comparison')
        result = {'case_id': case['case_id'], 'status': 'complete', 'session_id': analysis['session_id'],
                  'commitment': commitment, 'verdict': verdict,
                  'usage': {'analysis': analysis['usage'], 'commitment': committed['usage'],
                            'comparison': compared['usage']},
                  'data_unchanged': sha(root / 'workspace/data.csv') == data_sha, 'run_root': str(root)}
    except BaseException as error:
        result = {'case_id': case['case_id'], 'status': 'unavailable', 'error': str(error), 'run_root': str(root)}
    finally:
        remove_auth(root)
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_replays(workers):
    roster = json.loads((PILOT / 'roster.json').read_text())
    original = json.loads((RUNS / 'original/result.json').read_text())
    if original.get('status') != 'committed':
        raise SystemExit('Committed original required')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: run_replay(c, original['commitment']), roster['cases']))
    print(json.dumps({'complete': sum(r['status'] == 'complete' for r in results),
                      'unavailable': sum(r['status'] == 'unavailable' for r in results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('original', 'replays'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.action == 'original':
        print(json.dumps(run_original(), indent=2, ensure_ascii=False))
    else:
        run_replays(args.workers)
