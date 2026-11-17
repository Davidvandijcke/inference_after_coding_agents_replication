"""Prospective Linux genuine Codex CLI runner. Import is inert.

Selection uses native Codex tools. Commitment and comparison are separate turns
resumed by the recorded UUID, with comparison branches copied from the frozen
postcommit state. A relay capability is deliberately not an upstream API key.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sqlite3
import tempfile
import subprocess
import time
import uuid
from urllib.parse import urlparse

_network_spec = importlib.util.spec_from_file_location('scientific_network_policy', Path(__file__).with_name('network_policy.py'))
_network = importlib.util.module_from_spec(_network_spec)
_network_spec.loader.exec_module(_network)

_boundary_spec = importlib.util.spec_from_file_location('scientific_linux_boundary', Path(__file__).with_name('linux_boundary.py'))
_boundary = importlib.util.module_from_spec(_boundary_spec)
_boundary_spec.loader.exec_module(_boundary)

CLI = Path('<cluster-project>/luna-immigration-selection-20261015/runtime-0.157.0/codex-x86_64-unknown-linux-musl')
CODE_MODE_HOST = Path('<cluster-project>/luna-immigration-selection-20261015/runtime-0.157.0/codex-code-mode-host-x86_64-unknown-linux-musl')
GUEST_PATHS = _boundary.GUEST_PATHS
GUEST_CLI = _boundary.GUEST_CLI
CLI_VERSION = 'codex-cli 0.157.0'
MODEL = 'gpt-6-luna'
REASONING = 'high'
SERVICE_TIER = 'default'  # Responses spelling for standard tier; never priority.
DISABLED_FEATURES = ('apps', 'auth_elicitation', 'browser_use', 'browser_use_external',
    'browser_use_full_cdp_access', 'computer_use', 'fast_mode',
    'goals', 'guardian_approval', 'hooks', 'image_generation', 'in_app_browser',
    'in_app_chat', 'in_app_dictation', 'in_app_updates', 'memories', 'multi_agent',
    'multi_agent_v2', 'personality', 'plugin_sharing', 'plugins', 'remote_compaction_v2',
    'remote_plugin', 'shell_snapshot', 'skill_mcp_dependency_install', 'skill_search',
    'tool_call_mcp_elicitation', 'tool_suggest', 'unbounded_connection_retries',
    'view_image', 'workspace_dependencies')
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
COMMITMENT_SCHEMA = {'type': 'object', 'properties': {x: {'type': 'string'} for x in FIELDS},
                     'required': list(FIELDS), 'additionalProperties': False}
COMPARISON_SCHEMA = {'type': 'object', 'properties': {'model_null_match': {'type': 'boolean'},
    'reason': {'type': 'string'}}, 'required': ['model_null_match', 'reason'], 'additionalProperties': False}
INITIAL_STATE_SOURCE_FILES = {
    'runner_sha256': str(Path(__file__).resolve()),
    'network_policy_sha256': str(Path(__file__).with_name('network_policy.py')),
    'commitment_prompt_sha256': str(Path(__file__).with_name('commitment_prompt.txt')),
    'comparison_prompt_sha256': str(Path(__file__).with_name('comparison_prompt.txt')),
    'linux_boundary_sha256': str(Path(__file__).with_name('linux_boundary.py'))}


def sha_file(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError(f'Regular file required: {p}')
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def inventory(root, allow_cli_links=False):
    root = Path(root)
    out = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            if allow_cli_links and (p.resolve() == CLI or os.readlink(p) == GUEST_CLI):
                out[p.relative_to(root).as_posix()] = 'cli-symlink:' + os.readlink(p)
                continue
            raise ValueError(f'Symlink in frozen state: {p}')
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha_file(p)
    return out


def write_once(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write('\n')


def checked_uuid(value):
    value = str(value)
    if str(uuid.UUID(value)) != value:
        raise ValueError('Exact canonical session UUID required')
    return value


def relay_port(base_url):
    u = urlparse(base_url)
    if (u.scheme != 'http' or u.hostname != '127.0.0.1' or not u.port or
            u.path != '/v1' or u.username or u.password or u.query or u.fragment):
        raise ValueError('Relay must be explicit http://127.0.0.1:PORT/v1')
    return u.port


def _runtime(case_root=None, runtime_receipt=None):
    if runtime_receipt is None and case_root is not None:
        operator_path = Path(case_root) / 'operator.json'
        if operator_path.exists():
            runtime_receipt = json.loads(operator_path.read_text())['runtime_receipt']
    if runtime_receipt is None:
        raise ValueError('Trusted Linux runtime receipt required')
    receipt = _boundary.load_runtime_receipt(runtime_receipt)
    if Path(receipt['host_cli']).resolve() != CLI.resolve():
        raise ValueError('Runtime receipt points to another host CLI')
    if Path(receipt['host_code_mode_host']).resolve() != CODE_MODE_HOST.resolve():
        raise ValueError('Runtime receipt points to another code-mode host')
    return receipt


def config_text(base_url, case_root=None, network_profile="public_internet", *,
                runtime_receipt=None, developer_instructions_extra=None):
    relay_port(base_url)
    _network.checked(network_profile)
    runtime = _runtime(case_root, runtime_receipt)
    # Fresh CODEX_HOME contains this single controlled config and no auth, MCP,
    # plugin, memory, skill, hook, or inherited personal configuration.
    values = {
        'model': MODEL, 'model_provider': 'scientific_relay',
        'model_reasoning_effort': REASONING, 'service_tier': SERVICE_TIER,
        'approval_policy': 'never', 'sandbox_mode': 'danger-full-access',
        'web_search': 'disabled', 'project_doc_max_bytes': 0,
        'project_doc_fallback_filenames': [],
        'shell_environment_policy.inherit': 'none',
        'shell_environment_policy.experimental_use_profile': False,
        'shell_environment_policy.ignore_default_excludes': False,
        'memories.generate_memories': False, 'memories.use_memories': False,
        'model_providers.scientific_relay.name': 'Frozen scientific billing relay',
        'model_providers.scientific_relay.base_url': base_url,
        'model_providers.scientific_relay.wire_api': 'responses',
        'model_providers.scientific_relay.env_key': 'CODEX_RELAY_TOKEN',
        'model_providers.scientific_relay.requires_openai_auth': False,
        'model_providers.scientific_relay.supports_websockets': False,
        'model_providers.scientific_relay.request_max_retries': 2,
        'model_providers.scientific_relay.stream_max_retries': 2,
        'model_providers.scientific_relay.http_headers.X-Scientific-Service-Tier': 'default',
    }
    if developer_instructions_extra is not None:
        if (network_profile != 'public_internet' or
            not isinstance(developer_instructions_extra, str)):
            raise ValueError('Public-internet persona must be a string (empty neutral allowed)')
    if network_profile == "public_internet":
        values["developer_instructions"] = _network.DATA_SCOPE + (
            '\n\n' + developer_instructions_extra if developer_instructions_extra is not None else '')
    if case_root is not None:
        env = clean_environment(case_root, 'unused', runtime_receipt=runtime)
        for key, value in env.items():
            if key not in ('CODEX_HOME', 'CODEX_RELAY_TOKEN'):
                values[f'shell_environment_policy.set.{key}'] = value
    values.update({f'features.{name}': False for name in DISABLED_FEATURES})
    values['features.code_mode_host'] = True
    values['features.shell_tool'] = True
    values['features.unified_exec'] = True
    return '\n'.join(f'{k} = {json.dumps(v)}' for k, v in sorted(values.items())) + '\n'


def clean_environment(case_root, relay_token, *, runtime_receipt=None):
    runtime = _runtime(case_root, runtime_receipt)
    values = dict(runtime['guest_runtime_env'])
    values.update({'HOME': GUEST_PATHS['home'], 'CODEX_HOME': GUEST_PATHS['codex-home'],
        'TMPDIR': GUEST_PATHS['tmp'], 'TMPPREFIX': GUEST_PATHS['tmp'] + '/zsh',
        'CODEX_RELAY_TOKEN': relay_token,
        'R_ENVIRON_USER': '/dev/null', 'R_PROFILE_USER': '/dev/null',
        'R_ENVIRON': '/dev/null', 'R_PROFILE': '/dev/null',
        'PYTHONNOUSERSITE': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    return values


def initial_permission_policy(case_root, base_url, network_profile=None, packet_names=None, *, runtime_receipt=None):
    root = Path(case_root).resolve()
    if network_profile is None:
        op = json.loads((root / 'operator.json').read_text())
        network_profile = op['network_profile']
        packet_names = op['packet_sha256']
    runtime = _runtime(root, runtime_receipt)
    return _boundary.initial_permission_policy(root, base_url, network_profile,
                                               packet_names or {}, runtime)


def outer_policy(case_root, base_url, *, schema_file=None, network_profile=None,
                 packet_names=None, runtime_receipt=None):
    # Snapshot helper normalizes this deterministic policy object. A schema
    # file is an invocation-only read mount and never part of the cold policy.
    return initial_permission_policy(case_root, base_url, network_profile,
                                     packet_names, runtime_receipt=runtime_receipt)


def outer_command(case_root, base_url, guest_command, *, relay_token=None,
                  schema_file=None, extra_readonly=None, extra_writable=None):
    policy = outer_policy(case_root, base_url)
    return _boundary.outer_command(policy, guest_command, relay_token=relay_token,
        schema_file=schema_file, extra_readonly=extra_readonly,
        extra_writable=extra_writable)


def cli_command(case_root, *, session_id=None, schema_file=None):
    """No --last, ephemeral, fresh-judge session, or full-access bypass."""
    command = [GUEST_CLI, 'exec', '--cd', GUEST_PATHS['workspace'], '--strict-config', '--ignore-rules',
               '--model', MODEL, '--skip-git-repo-check']
    if session_id is not None:
        command += ['resume', checked_uuid(session_id)]
    command += ['--json']
    if schema_file is not None:
        command += ['--output-schema', _boundary.GUEST_SCHEMA]
    return command + ['-']


def initialize_case(root, packet, research_prompt, base_url, limits, *,
                    network_profile="public_internet", runtime_receipt=None,
                    developer_instructions_extra=None):
    """Freeze a fresh public packet; target and commitment are not inputs here."""
    root = Path(root).resolve()
    _network.checked(network_profile)
    runtime = _runtime(runtime_receipt=runtime_receipt)
    curated_verification = _boundary.verify_curated_prefix_full(runtime)
    if root.exists():
        raise ValueError('Fresh task root required; no resume/restart by directory reuse')
    required = {'wall_seconds', 'output_reservation_tokens'}
    if set(limits) != required or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError('Explicit positive frozen limits required')
    root.mkdir(mode=0o700)
    for name in ('workspace', 'home', 'codex-home', 'tmp'):
        (root / name).mkdir(mode=0o700)
    for name, source in packet.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or len(p.parts) != 1 or p.name.startswith('.') or p.name.startswith('AGENTS'):
            raise ValueError('Public packet uses plain top-level approved filenames only')
        sha_file(source)
        shutil.copyfile(source, root / 'workspace' / name)
    text = config_text(base_url, root, network_profile, runtime_receipt=runtime,
                       developer_instructions_extra=developer_instructions_extra)
    (root / 'codex-home/config.toml').write_text(text)
    operator = {'schema': 'codex-cli-scientific-operator-v1',
        'cli_path': str(CLI), 'guest_cli_path': GUEST_CLI,
        'cli_sha256': runtime['cli_sha256'], 'cli_version': CLI_VERSION,
        'code_mode_host_path': str(CODE_MODE_HOST),
        'guest_code_mode_host_path': _boundary.GUEST_CODE_MODE_HOST,
        'code_mode_host_sha256': runtime['code_mode_host_sha256'],
        'model': MODEL, 'reasoning_effort': REASONING, 'service_tier': SERVICE_TIER,
        'config_sha256': sha_file(root / 'codex-home/config.toml'),
        'initial_prompt': research_prompt, 'initial_prompt_sha256': hashlib.sha256(research_prompt.encode()).hexdigest(),
        'packet_sha256': inventory(root / 'workspace'), 'limits': limits, 'relay_base_url': base_url,
        'network_profile': network_profile, 'network_policy_sha256': sha_file(Path(__file__).with_name('network_policy.py')),
        'linux_boundary_sha256': sha_file(Path(__file__).with_name('linux_boundary.py')),
        'runtime_receipt': runtime, 'runtime_receipt_hash': runtime['hash'],
        'outer_policy_sha256': digest(outer_policy(root, base_url, network_profile=network_profile,
            packet_names=packet, runtime_receipt=runtime)),
        'sandbox_mode': 'linux_bwrap_guest_fixed_paths_public_host_network',
        'runner_sha256': sha_file(__file__),
        'commitment_prompt_sha256': sha_file(Path(__file__).with_name('commitment_prompt.txt')),
        'comparison_prompt_sha256': sha_file(Path(__file__).with_name('comparison_prompt.txt')),
        'status': 'unactivated_requires_funded_relay_and_offline_acceptance'}
    if developer_instructions_extra is not None:
        operator['developer_instructions_extra'] = developer_instructions_extra
        operator['developer_instructions_extra_sha256'] = hashlib.sha256(
            developer_instructions_extra.encode()).hexdigest()
    if curated_verification is not None:
        operator['curated_prefix_verification'] = curated_verification
    operator['hash'] = digest(operator)
    # Metadata is kept outside every path visible to the agent.
    write_once(root / 'operator.json', operator)
    return operator


def parse_events(stdout, expected_session=None):
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    sessions = {x['thread_id'] for x in events if x.get('type') == 'thread.started'}
    if len(sessions) != 1:
        raise ValueError('Exactly one explicit started session receipt required')
    session = checked_uuid(next(iter(sessions)))
    if expected_session is not None and session != checked_uuid(expected_session):
        raise ValueError('Resume changed session ID')
    # The CLI emits recoverable reconnect notices as top-level error events.
    # Accept only that known notice, before a successful terminal turn receipt.
    # A failed turn, unknown error, or unfinished subsequent turn stays unavailable.
    completed = [i for i, x in enumerate(events) if x.get('type') == 'turn.completed']
    lifecycle = [x.get('type') for x in events
                 if x.get('type') in ('turn.started', 'turn.completed', 'turn.failed')]
    reconnect = re.compile(r'Reconnecting\.\.\. [1-9][0-9]*/[1-9][0-9]* \([^\r\n]+\)')
    if (not completed or lifecycle[-1] != 'turn.completed'
            or any(x.get('type') == 'turn.failed' for x in events)
            or any(x.get('type') == 'error' and
                   (i >= completed[-1] or not isinstance(x.get('message'), str)
                    or reconnect.fullmatch(x['message']) is None)
                   for i, x in enumerate(events))):
        raise ValueError('CLI turn unavailable; retain raw events and spending liabilities')
    answers = [x['item']['text'] for x in events if x.get('type') == 'item.completed'
               and x.get('item', {}).get('type') == 'agent_message']
    if not answers:
        raise ValueError('Missing final agent message')
    return {'session_id': session, 'answer': answers[-1], 'events': events}


def freeze_commitment_snapshot(case_root, session_id, commitment):
    root = Path(case_root).resolve()
    checked_uuid(session_id)
    if set(commitment) != set(FIELDS) or any(not isinstance(v, str) or not v.strip() for v in commitment.values()):
        raise ValueError('Malformed declaration; no semantic/code audit or repair')
    snapshot = root / 'frozen-commitment'
    snapshot.mkdir()
    for name in ('workspace', 'codex-home', 'home', 'tmp'):
        shutil.copytree(root / name, snapshot / name, symlinks=True)
    manifest = {'schema': 'codex-cli-committed-prefix-v1', 'session_id': session_id,
        'commitment': commitment, 'commitment_hash': digest(commitment),
        'operator_hash': json.loads((root / 'operator.json').read_text())['hash'],
        'files': inventory(snapshot, allow_cli_links=True)}
    manifest['hash'] = digest(manifest)
    write_once(root / 'commitment.json', manifest)
    return manifest


def restore_comparison_branch(case_root, target_id):
    """Sequential branches preserve the original absolute cwd and session UUID."""
    root = Path(case_root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(target_id)):
        raise ValueError('Invalid target ID')
    manifest = json.loads((root / 'commitment.json').read_text())
    snapshot = root / 'frozen-commitment'
    if inventory(snapshot, allow_cli_links=True) != manifest['files']:
        raise ValueError('Frozen commitment prefix changed')
    branch = root / 'branches' / str(target_id)
    branch.mkdir(parents=True, exist_ok=False)
    for name in ('workspace', 'codex-home', 'home', 'tmp'):
        shutil.rmtree(root / name)
        shutil.copytree(snapshot / name, root / name, symlinks=True)
    write_once(branch / 'start.json', {'session_id': manifest['session_id'], 'target_id': str(target_id),
        'committed_prefix_hash': manifest['hash'], 'restored_files': inventory(root / 'codex-home', allow_cli_links=True)})
    return manifest['session_id'], branch


def invoke_offline(case_root, prompt, relay_token, output_root, *, session_id=None, schema=None, timeout=30, funded_binding=None):
    """Free acceptance only: caller supplies a local mock relay, never upstream auth.

Paid invocation is intentionally not exported until the independent relay's
funding/lease receipt and sandbox acceptance are integrated and reviewed.
"""
    root = Path(case_root).resolve(); output = Path(output_root).resolve()
    if os.uname().sysname != 'Linux':
        raise RuntimeError('Linux bubblewrap runner cannot run on another host')
    operator = json.loads((root / 'operator.json').read_text())
    if sha_file(Path(__file__).with_name('network_policy.py')) != operator['network_policy_sha256']:
        raise ValueError('Frozen network policy changed')
    if sha_file(Path(__file__).with_name('linux_boundary.py')) != operator['linux_boundary_sha256']:
        raise ValueError('Frozen Linux boundary changed')
    runtime = _runtime(root)
    if runtime['hash'] != operator['runtime_receipt_hash']:
        raise ValueError('Frozen Linux runtime receipt changed')
    curated_verification = _boundary.verify_curated_prefix_full(runtime)
    if curated_verification != operator.get('curated_prefix_verification'):
        raise ValueError('Frozen curated Python prefix differs before native phase')
    if sha_file(CLI) != operator['cli_sha256'] or sha_file(root / 'codex-home/config.toml') != operator['config_sha256']:
        raise ValueError('Frozen CLI/config changed')
    policy = outer_policy(root, operator['relay_base_url'])
    if digest(policy) != operator['outer_policy_sha256']:
        raise ValueError('Frozen Linux mount/network policy changed')
    activation = None
    if funded_binding is not None:
        activation = validate_funded_binding(root, funded_binding)
        expected_token = Path(activation['dummy_token_file']).read_text().strip()
        if relay_token != expected_token:
            raise ValueError('Scoped relay token differs from funded manifest')
    elif not relay_token.startswith('offline-fixture-'):
        raise ValueError('A funded relay manifest binding is required')
    output.mkdir(parents=True, exist_ok=False)
    schema_file = output / 'schema.json' if schema is not None else None
    if schema_file: write_once(schema_file, schema)
    command = cli_command(root, session_id=session_id, schema_file=schema_file)
    wrapped = _boundary.outer_command(policy, command, relay_token=relay_token,
                                      schema_file=schema_file)
    proc = subprocess.Popen(wrapped,
        cwd=root, env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent'}, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(prompt, timeout=min(timeout, operator['limits']['wall_seconds']))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    finally:
        # Close managed tool subprocesses; never use session files from a live CLI.
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    (output / 'events.jsonl').write_text(stdout)
    (output / 'stderr.txt').write_text(stderr)
    inventory_error = None
    try:
        state_files = inventory(root / 'codex-home', allow_cli_links=True)
        workspace_files = inventory(root / 'workspace')
        if any(workspace_files.get(name) != sha for name, sha in operator['packet_sha256'].items()):
            raise ValueError('Supplied research packet changed; preserve unavailable outcome')
    except Exception as e:
        inventory_error = str(e); state_files = None; workspace_files = None
    receipt = {'schema': 'codex-cli-linux-phase-v1', 'operator_hash': operator['hash'],
        'argv': command, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'guest_paths': GUEST_PATHS, 'guest_cli': GUEST_CLI,
        'outer_policy_sha256': operator['outer_policy_sha256'],
        'runtime_receipt_hash': runtime['hash'],
        'curated_prefix_verification': curated_verification,
        'expected_session_id': session_id, 'returncode': proc.returncode, 'timed_out': timed_out,
        'events_sha256': sha_file(output / 'events.jsonl'), 'stderr_sha256': sha_file(output / 'stderr.txt'),
        'state_sha256': state_files, 'workspace_sha256': workspace_files, 'artifact_inventory_error': inventory_error,
        'mode': 'funded_relay' if activation else 'offline_mock_only', 'paid_activation_enabled': bool(activation),
        'funded_binding_hash': activation['hash'] if activation else None,
        'allocation_hash': activation['allocation_hash'] if activation else None,
        'relay_ledger': activation['ledger'] if activation else None}
    receipt['hash'] = digest(receipt);write_once(output / 'phase.json', receipt)
    if proc.returncode != 0 or timed_out or inventory_error:
        clean = lambda value: value.replace(relay_token, '<scoped-dummy-token>')[-1800:]
        raise RuntimeError('CLI turn unavailable: returncode=' + str(proc.returncode) +
            ' timed_out=' + str(timed_out) + ' inventory_error=' + str(inventory_error) +
            ' stderr_tail=' + repr(clean(stderr)) + ' events_tail=' + repr(clean(stdout)) +
            ' receipts=' + str(output))
    return parse_events(stdout, session_id), receipt


def run_analysis(case_root, relay_token, output_root, *, funded_binding=None):
    """Native session: native CLI analysis; initial prompt is unchanged."""
    operator = json.loads((Path(case_root) / 'operator.json').read_text())
    return invoke_offline(case_root, operator['initial_prompt'], relay_token, output_root,
                          timeout=operator['limits']['wall_seconds'], funded_binding=funded_binding)


def run_commitment(case_root, analysis_session_id, relay_token, output_root, *, funded_binding=None):
    """Native session: post-analysis declaration in that exact CLI session."""
    prompt = Path(__file__).with_name('commitment_prompt.txt').read_text()
    result, receipt = invoke_offline(case_root, prompt, relay_token, output_root,
        session_id=checked_uuid(analysis_session_id), schema=COMMITMENT_SCHEMA,
        timeout=json.loads((Path(case_root)/'operator.json').read_text())['limits']['wall_seconds'], funded_binding=funded_binding)
    commitment = json.loads(result['answer'])
    frozen = freeze_commitment_snapshot(case_root, result['session_id'], commitment)
    return {'session_id': result['session_id'], 'commitment': commitment,
            'committed_prefix': frozen, 'phase_receipt': receipt}


def run_comparison(case_root, target_id, target_commitment, relay_token, output_root, *, funded_binding=None):
    """Native session: copy frozen prefix, resume same UUID, trust its label."""
    if set(target_commitment) != set(FIELDS) or any(not isinstance(x,str) or not x for x in target_commitment.values()):
        raise ValueError('Target must be the original declared substantive commitment')
    session_id, branch = restore_comparison_branch(case_root, target_id)
    template = Path(__file__).with_name('comparison_prompt.txt').read_text()
    target_json = json.dumps(target_commitment,ensure_ascii=False,separators=(',',':'))
    prompt = template.replace('{{TARGET_JSON}}',target_json)
    result, receipt = invoke_offline(case_root,prompt,relay_token,output_root,
        session_id=session_id,schema=COMPARISON_SCHEMA,
        timeout=json.loads((Path(case_root)/'operator.json').read_text())['limits']['wall_seconds'], funded_binding=funded_binding)
    comparison = json.loads(result['answer'])
    if (set(comparison) != {'model_null_match','reason'} or type(comparison['model_null_match']) is not bool
            or not isinstance(comparison['reason'],str) or not comparison['reason']):
        raise ValueError('Malformed comparison remains unavailable; no label repair')
    outcome = {'session_id':session_id,'target_id':str(target_id),'target_commitment_hash':digest(target_commitment),
        'comparison':comparison,'phase_receipt':receipt}
    write_once(branch/'result.json',outcome)
    return outcome


def verify_hidden_paths(case_root, host_paths):
    """Free canary: guest cannot read trusted key or curator host paths."""
    root = Path(case_root).resolve()
    op = json.loads((root / 'operator.json').read_text())
    if not host_paths:
        raise ValueError('At least one trusted host path canary required')
    paths = [str(Path(p).resolve()) for p in host_paths]
    code = ('import pathlib\n'
        'paths = ' + repr(paths) + '\n'
        'for path in paths:\n'
        '    try:\n'
        '        pathlib.Path(path).read_bytes()\n'
        '    except (PermissionError, FileNotFoundError):\n'
        '        continue\n'
        '    raise AssertionError("HOST_READ_LEAK:" + path)\n'
        'print("TRUSTED_AND_CURATOR_HOST_PATHS_HIDDEN")\n')
    policy = outer_policy(root, op['relay_base_url'])
    runtime = _runtime(root)
    guest_python = runtime.get('guest_python')
    if not guest_python or not guest_python.startswith('/sw/'):
        raise ValueError('Module Python path missing from trusted runtime receipt')
    wrapped = _boundary.outer_command(policy, [guest_python, '-c', code])
    result = subprocess.run(wrapped, cwd=root, env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent'},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
    if result.returncode or result.stdout.strip() != 'TRUSTED_AND_CURATOR_HOST_PATHS_HIDDEN':
        raise RuntimeError('Linux host-path canary failed: ' + result.stderr[-1000:])
    proof = {'schema': 'codex-cli-linux-read-boundary-canary-v1',
        'tested_host_paths': paths, 'read_denied': True, 'provider_calls': 0,
        'outer_policy_sha256': op['outer_policy_sha256'],
        'stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest()}
    proof['hash'] = digest(proof)
    return proof


def verify_trusted_directory(case_root, launch_manifest):
    root = Path(case_root).resolve()
    launch = json.loads(Path(launch_manifest).read_text())
    trusted = Path(launch['real_key_file']).resolve().parent
    fd, canary = tempfile.mkstemp(prefix='cli-read-denial-canary-', dir=trusted)
    os.write(fd, b'NONSECRET_ISOLATION_CANARY'); os.close(fd)
    try:
        proof = verify_hidden_paths(root, [canary])
        proof['trusted_directory'] = str(trusted)
        proof['real_key_read'] = False
        proof['hash'] = digest({k: v for k, v in proof.items() if k != 'hash'})
        write_once(root / 'trusted-boundary-canary.json', proof)
        return proof
    finally:
        Path(canary).unlink()


def bind_funded_relay(case_root, launch_manifest, expected_allocation_hash):
    root=Path(case_root).resolve();manifest=Path(launch_manifest).resolve()
    launch=json.loads(manifest.read_text());op=json.loads((root/'operator.json').read_text())
    if (launch.get('schema') not in ('codex-cli-luna-smoke-relay-launch-v1',
                                    'codex-cli-luna-network-pilot-relay-launch-v1',
                                    'codex-cli-luna-linux-relay-launch-v1')
        or launch.get('bind_host')!='127.0.0.1'
        or launch.get('allocation_sha256')!=expected_allocation_hash
        or op['relay_base_url']!=f"http://127.0.0.1:{launch['port']}/v1"):
        raise ValueError('Wrong funded campaign/relay/endpoint binding')
    metadata=json.loads(Path(launch['allocation_metadata_file']).read_text())
    pricing=json.loads(Path(launch['pricing_file']).read_text())
    if (metadata['allocation_sha256']!=expected_allocation_hash or metadata['cap_usd']!=launch['allocation_cap_usd']
        or metadata['model']!=MODEL or metadata['reasoning_effort']!=REASONING or metadata['service_tier']!=SERVICE_TIER
        or pricing['max_output_tokens']!=op['limits']['output_reservation_tokens']):
        raise ValueError('Funded operator/price reservation differs')
    with sqlite3.connect(f"file:{launch['ledger']}?mode=ro",uri=True) as db:
        ledger_metadata=dict(db.execute('SELECT key,value FROM metadata'))
        if (float(ledger_metadata['cap_usd'])!=float(metadata['cap_usd'])
            or json.loads(ledger_metadata['allocation_metadata_json'])!=metadata
            or json.loads(ledger_metadata['pricing_document_json'])!=pricing):
            raise ValueError('Actual child ledger cap/allocation/pricing differs')
    canary=verify_trusted_directory(root,manifest)
    binding={'schema':'codex-cli-funded-relay-binding-v1','operator_hash':op['hash'],
        'manifest':str(manifest),'manifest_sha256':sha_file(manifest),'allocation_hash':expected_allocation_hash,
        'allocation_metadata_file':launch['allocation_metadata_file'],'allocation_metadata_sha256':sha_file(launch['allocation_metadata_file']),
        'pricing_file':launch['pricing_file'],'pricing_file_sha256':sha_file(launch['pricing_file']),
        'ledger':launch['ledger'],'cap_usd':metadata['cap_usd'],'dummy_token_file':launch['dummy_token_file'],
        'dummy_token_sha256':sha_file(launch['dummy_token_file']),'canary_hash':canary['hash'],
        'request_limit_policy':'Atomic funded spend cap in relay; no separate physical-request-count cap',
        'output_limit_policy':'Native CLI omission unchanged; relay reserves full128000-token output window'}
    binding['hash']=digest(binding);write_once(root/'funded-relay-binding.json',binding)
    return root/'funded-relay-binding.json'


def validate_funded_binding(case_root, binding_file):
    root=Path(case_root).resolve();binding=json.loads(Path(binding_file).read_text());claimed=binding.pop('hash')
    if digest(binding)!=claimed:raise ValueError('Funded binding self hash differs')
    binding['hash']=claimed;op=json.loads((root/'operator.json').read_text())
    if binding['operator_hash']!=op['hash']:raise ValueError('Funded binding belongs to another operator')
    for path_key,sha_key in [('manifest','manifest_sha256'),('allocation_metadata_file','allocation_metadata_sha256'),
                             ('pricing_file','pricing_file_sha256'),('dummy_token_file','dummy_token_sha256')]:
        if sha_file(binding[path_key])!=binding[sha_key]:raise ValueError('Trusted relay metadata changed')
    return binding
