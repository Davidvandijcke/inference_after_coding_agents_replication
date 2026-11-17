"""Prospective pre-analysis state capture and replay cloning. No CLI/model calls.

Only a cold, newly initialized native case is eligible. This does not reconstruct
old pre-analysis states. Snapshot/clone receipts live outside agent-readable trees.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess

TREES = ('workspace', 'home', 'codex-home', 'tmp')
CONFIG = 'codex-home/config.toml'
DATA = 'workspace/data.csv'
PATH_KEYS = {'shell_environment_policy.set.HOME', 'shell_environment_policy.set.TMPDIR',
             'shell_environment_policy.set.TMPPREFIX'}
RELAY_KEY = 'model_providers.scientific_relay.base_url'
LINUX_GUEST_PATHS = {'workspace': '/workspace', 'home': '/home/researcher',
                     'tmp': '/tmp', 'codex-home': '/codex-home'}


def _linux(runner):
    paths = getattr(runner, 'GUEST_PATHS', None)
    if paths is not None and paths != LINUX_GUEST_PATHS:
        raise ValueError('Linux runner must expose the approved neutral guest paths')
    return paths is not None


def _source_files(runner):
    files = {'runner_sha256': Path(runner.__file__),
             'network_policy_sha256': Path(runner.__file__).with_name('network_policy.py'),
             'commitment_prompt_sha256': Path(runner.__file__).with_name('commitment_prompt.txt'),
             'comparison_prompt_sha256': Path(runner.__file__).with_name('comparison_prompt.txt')}
    files.update(getattr(runner, 'INITIAL_STATE_SOURCE_FILES', {}))
    return {key: Path(value) for key, value in files.items()}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _load_runner(runner):
    if runner is not None and hasattr(runner, 'initialize_case'):
        return runner
    path = Path(runner) if runner is not None else Path(__file__).with_name('runner.py')
    spec = importlib.util.spec_from_file_location('initial_state_scientific_runner', str(path.resolve()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path):
    x = json.loads(Path(path).read_text())
    value = dict(x); claimed = value.pop('hash', None)
    if claimed != _hash(value):
        raise ValueError('Self hash differs: ' + str(path))
    return x


def _write_json(path, x):
    x = dict(x); x['hash'] = _hash(x)
    with Path(path).open('x') as f:
        json.dump(x, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    return x


def _safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or str(p) != value:
        raise ValueError('Unsafe snapshot path')
    return p


def _opaque_root(root, neutral_parent, guest_paths=False):
    root = Path(root).absolute()
    parent = Path(neutral_parent).resolve()
    if root.parent.resolve() != parent or not re.fullmatch(r'[0-9a-f]{16,64}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', root.name):
        raise ValueError('Original and replay roots must use the same neutral parent and opaque hexadecimal/UUID names')
    if not guest_paths and re.search(r'original|replay|target|stratum|hypothesis|dgp|null|immigration|castle|card|results', str(parent), re.I):
        raise ValueError('Agent-visible parent path contains scientific or role labels')
    if root.is_symlink():
        raise ValueError('Case root cannot be a symlink')
    return parent / root.name


def _inventory(root):
    root = Path(root)
    out = {}
    for tree in TREES:
        p = root / tree
        if not p.is_dir() or p.is_symlink():
            raise ValueError('Required regular initial directory missing: ' + tree)
        for item in [p] + sorted(p.rglob('*')):
            info = item.lstat(); relative = item.relative_to(root).as_posix()
            _safe_relative(relative)
            if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError('Links or special files forbidden in initial state: ' + relative)
            value = {'kind': 'directory' if item.is_dir() else 'file', 'mode': stat.S_IMODE(info.st_mode)}
            if item.is_file():
                value.update(size=info.st_size, sha256=_sha(item))
            out[relative] = value
    return out


def _config_values(text):
    values = {}
    for line in text.splitlines():
        key, value = line.split(' = ', 1)
        if key in values:
            raise ValueError('Duplicate initial config key')
        values[key] = json.loads(value)
    return values


def _normalized_config(text, root, base_url, runner=None):
    values = _config_values(text)
    if runner is not None and _linux(runner):
        required = {'shell_environment_policy.set.HOME': '/home/researcher',
                    'shell_environment_policy.set.TMPDIR': '/tmp',
                    'shell_environment_policy.set.TMPPREFIX': '/tmp/zsh'}
        if any(values.get(key) != value for key, value in required.items()):
            raise ValueError('Linux config exposes nonneutral guest home/temp paths')
        key = 'shell_environment_policy.set.CODEX_HOME'
        if key in values and values[key] != '/codex-home':
            raise ValueError('Linux config exposes nonneutral Codex home')
        if values[RELAY_KEY] != base_url:
            raise ValueError('Unexpected relay binding')
        values[RELAY_KEY] = '<BILLING_RELAY>'
        return values
    for key in PATH_KEYS:
        value = values[key]
        if not value.startswith(str(root) + '/'):
            raise ValueError('Unexpected case-specific environment path')
        values[key] = '<CASE_ROOT>' + value[len(str(root)):]
    if values[RELAY_KEY] != base_url:
        raise ValueError('Unexpected relay binding')
    values[RELAY_KEY] = '<BILLING_RELAY>'
    return values


def _normalized_policy(text, root, base_url, runner):
    if _linux(runner):
        def normalize(value):
            if isinstance(value, dict): return {k: normalize(v) for k, v in value.items()}
            if isinstance(value, list): return [normalize(v) for v in value]
            if isinstance(value, str):
                if value == str(root) or value.startswith(str(root) + '/'):
                    return '<CASE_ROOT>' + value[len(str(root)):]
                if value == base_url: return '<BILLING_RELAY>'
            return value
        return normalize(text)
    return text.replace(str(root), '<CASE_ROOT>').replace(
        'localhost:' + str(runner.relay_port(base_url)), 'localhost:<BILLING_PORT>')


def _policy(runner, root, base_url, operator):
    if _linux(runner):
        value = runner.initial_permission_policy(root, base_url, operator['network_profile'],
                                                 sorted(operator['packet_sha256']))
        if not isinstance(value, dict): raise ValueError('Linux permission policy must be a deterministic mapping')
        return value
    return runner.outer_policy(root, base_url)


def _policy_hash(value):
    return _hash(value) if isinstance(value, dict) else hashlib.sha256(value.encode()).hexdigest()


def _policy_differences(before, after, path='$'):
    if before == after: return {}
    if isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after):
        out = {}
        for key in before: out.update(_policy_differences(before[key], after[key], path + '.' + key))
        return out
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        out = {}
        for i, (left, right) in enumerate(zip(before, after)):
            out.update(_policy_differences(left, right, path + '[' + str(i) + ']'))
        return out
    return {path: {'before': before, 'after': after}}


def _linux_receipt(value, runner=None):
    if isinstance(value, (str, Path)):
        raw = json.loads(Path(value).read_text())
        if raw.get('schema') == 'greatlakes-native-runtime-environment-v1':
            if runner is None or not hasattr(runner, '_boundary'):
                raise ValueError('Root Linux environment receipt requires the approved Linux boundary loader')
            value = runner._boundary.load_runtime_receipt(value)
        else:
            value = _read_json(value)
    elif isinstance(value, dict) and value.get('schema') == 'greatlakes-native-runtime-environment-v1':
        if runner is None or not hasattr(runner, '_boundary'):
            raise ValueError('Root Linux environment receipt requires the approved Linux boundary loader')
        value = runner._boundary.load_runtime_receipt(value)
    if not isinstance(value, dict):
        raise ValueError('Linux requires the trusted current Linux runtime receipt')
    check = dict(value); claimed = check.pop('hash', None)
    scientific = value.get('schema') == 'scientific-linux-runtime-v1'
    if claimed != _hash(check) or (not scientific and value.get('platform') != 'Linux'):
        raise ValueError('Linux runtime receipt hash/platform differs; Mac fingerprints cannot be reused')
    if scientific:
        if '/sw' not in value.get('public_mounts', {}) or not all(k in value for k in
                ('host_cli', 'cli_sha256', 'host_code_mode_host',
                 'code_mode_host_sha256', 'bwrap', 'bwrap_sha256', 'guest_runtime_env')):
            raise ValueError('Incomplete canonical Linux runtime receipt')
        return value
    if not isinstance(value.get('files'), dict) or not value['files'] or '/sw' not in value.get('read_only_roots', []):
        raise ValueError('Linux receipt must declare file fingerprints and read-only /sw modules')
    return value


def _linux_public_metadata(receipt):
    """Metadata only, using the root-validated public module environment."""
    environment = dict(receipt['guest_runtime_env'])
    environment.update(HOME='/nonexistent', TMPDIR='/tmp', R_ENVIRON_USER='/dev/null',
        R_PROFILE_USER='/dev/null', R_ENVIRON='/dev/null', R_PROFILE='/dev/null', PYTHONNOUSERSITE='1')
    # These are guest-local paths, not shared host package stores.
    for key in ('PATH', 'R_LIBS_USER', 'PYTHONPATH'):
        if key in environment:
            environment[key] = ':'.join(p for p in environment[key].split(':') if p and not any(
                p == local or p.startswith(local + '/') for local in LINUX_GUEST_PATHS.values()))
    files = {}; metadata = {'modules_environment': receipt['guest_runtime_env']}
    tools = {}
    for name in ('Rscript', 'python3', 'git', 'clang'):
        found = shutil.which(name, path=environment['PATH'])
        if found is None:
            metadata[name] = {'state': 'missing'}; continue
        path = Path(found).resolve(); tools[name] = str(path)
        result = subprocess.run([str(path), '--version'], env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
        metadata[name] = {'path': str(path), 'version': (result.stdout + result.stderr).strip()}
        files['tool:' + name] = {'path': str(path), 'sha256': _sha(path)}
    if 'Rscript' in tools:
        code = ('p<-installed.packages();for(i in seq_len(nrow(p)))'
                'cat(p[i,"Package"],"\\t",file.path(p[i,"LibPath"],p[i,"Package"],"DESCRIPTION"),"\\n",sep="")')
        result = subprocess.run([tools['Rscript'], '--vanilla', '-e', code], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        if result.returncode: raise ValueError('Linux R metadata read failed')
        for line in result.stdout.splitlines():
            name, path = line.split('\t', 1)
            files['R:' + name] = {'path': str(Path(path).resolve()), 'sha256': _sha(path)}
    if 'python3' in tools:
        code = ('import importlib.metadata as m,json;'
                'print(json.dumps(sorted([(d.metadata.get("Name","unknown"),str(d._path)) for d in m.distributions()])))')
        result = subprocess.run([tools['python3'], '-I', '-c', code], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        if result.returncode: raise ValueError('Linux Python metadata read failed')
        for name, directory in json.loads(result.stdout):
            for filename in ('METADATA', 'PKG-INFO', 'RECORD'):
                path = Path(directory) / filename
                if path.is_file(): files['Python:' + name + '/' + filename] = {'path': str(path.resolve()), 'sha256': _sha(path)}
    roots = list(receipt['public_mounts'].values())
    if any(not any(entry['path'] == p or entry['path'].startswith(p.rstrip('/') + '/') for p in roots)
           for entry in files.values()):
        raise ValueError('Discovered Linux dependency is outside approved public read-only mounts')
    return files, metadata


def _linux_fingerprints(runner, root, extra_files, runtime_receipt):
    receipt = _linux_receipt(runtime_receipt, runner)
    files = {}
    if receipt.get('schema') == 'scientific-linux-runtime-v1':
        declared = {'cli': {'path': receipt['host_cli'], 'sha256': receipt['cli_sha256']},
                    'code-mode-host': {'path': receipt['host_code_mode_host'],
                                       'sha256': receipt['code_mode_host_sha256']},
                    'bwrap': {'path': receipt['bwrap'], 'sha256': receipt['bwrap_sha256']}}
        curated = receipt.get('curated_python_prefix')
        if curated is not None:
            declared['curated:prefix-receipt'] = {
                'path': curated['receipt_file'], 'sha256': curated['receipt_sha256']}
            declared['curated:site-files'] = {
                'path': curated['site_inventory'], 'sha256': curated['site_inventory_sha256']}
        public, metadata = _linux_public_metadata(receipt); declared.update(public)
        read_only_roots = list(receipt['public_mounts'])
    else:
        declared = dict(receipt['files']); metadata = receipt.get('metadata', {})
        read_only_roots = receipt['read_only_roots']
    for key, path in _source_files(runner).items():
        declared['source:' + key] = {'path': str(path), 'sha256': _sha(path)}
    for key, path in (extra_files or {}).items():
        declared['extra:' + key] = {'path': str(path), 'sha256': _sha(path)}
    for name, entry in sorted(declared.items()):
        path = Path(entry['path']).resolve()
        if Path(root).resolve() in path.parents:
            raise ValueError('Case-local dependencies belong in the initial tree, not the Linux shared receipt')
        if not path.is_file() or _sha(path) != entry['sha256']:
            raise ValueError('Linux shared runtime file changed: ' + name)
        mode = stat.S_IMODE(path.stat().st_mode)
        if 'mode' in entry and entry['mode'] != mode:
            raise ValueError('Linux shared runtime file mode changed: ' + name)
        files[name] = {'path': str(path), 'sha256': entry['sha256'], 'mode': mode}
    return {'platform': 'Linux', 'receipt_hash': receipt['hash'], 'files': files, 'metadata': metadata,
            'read_only_roots': sorted(read_only_roots),
            'scope': 'Trusted Linux environment receipt and declared files only; no Mac discovery, original HOME access, or whole-host byte claim.'}


def _shared_fingerprints(runner, root, extra_files, runtime_receipt=None):
    """Hash narrow shared binaries/package metadata; never copy the host tree."""
    if _linux(runner):
        return _linux_fingerprints(runner, root, extra_files, runtime_receipt)
    base = Path(runner.__file__).resolve().parent
    paths = {'runner': Path(runner.__file__).resolve(), 'cli': Path(runner.CLI).resolve()}
    for name in ('network_policy.py', 'commitment_prompt.txt', 'comparison_prompt.txt'):
        paths[name] = base / name
    environment = runner.clean_environment(root, 'metadata-only-no-provider')
    environment.pop('CODEX_RELAY_TOKEN', None)
    # Shared-store discovery must not read the original's post-analysis home.
    # Case-local packages are already preserved byte-for-byte in the snapshot.
    environment.update(HOME='/nonexistent', TMPDIR='/private/tmp', TMPPREFIX='/private/tmp/zsh')
    packages = {}
    for tool in ('Rscript', 'python3', 'git', 'clang'):
        found = shutil.which(tool, path=environment['PATH'])
        packages[tool] = {'state': 'missing'} if found is None else {'state': 'present'}
        if found is not None:
            paths['tool:' + tool] = Path(found).resolve()
            version = subprocess.run([str(paths['tool:' + tool]), '--version'], env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
            packages[tool]['version_output'] = (version.stdout + version.stderr).strip()
    rscript = paths.get('tool:Rscript')
    if rscript:
        code = ('p<-installed.packages();for(i in seq_len(nrow(p)))'
                'cat(p[i,"Package"],"\\t",file.path(p[i,"LibPath"],p[i,"Package"]),"\\n",sep="")')
        result = subprocess.run([str(rscript), '--vanilla', '-e', code], env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        if result.returncode:
            raise ValueError('Cannot fingerprint initial R packages: ' + result.stderr[-500:])
        for line in result.stdout.splitlines():
            name, directory = line.split('\t', 1)
            packages['R:' + name] = {'state': 'present' if directory else 'missing'}
            if directory:
                folder = Path(directory)
                files = [folder / 'DESCRIPTION']
                if name in ('base', 'stats', 'utils', 'methods', 'fixest', 'sandwich', 'lmtest', 'jsonlite'):
                    files += [folder / 'NAMESPACE'] + sorted((folder / 'R').glob('*.rd?'))
                for f in files:
                    if f.is_file():
                        paths['R:' + name + '/' + f.relative_to(folder).as_posix()] = f.resolve()
    python = paths.get('tool:python3')
    if python:
        code = ('import importlib.metadata as m,json;'
                'print(json.dumps(sorted([(d.metadata.get("Name","unknown"),str(d._path)) for d in m.distributions()])))')
        result = subprocess.run([str(python), '-I', '-c', code], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        if result.returncode:
            raise ValueError('Cannot fingerprint initial Python distribution metadata')
        for name, directory in json.loads(result.stdout):
            folder = Path(directory)
            packages['Python:' + name] = {'state': 'present', 'metadata_directory': str(folder)}
            for namepart in ('METADATA', 'PKG-INFO', 'RECORD'):
                f = folder / namepart
                if f.is_file():
                    paths['Python:' + name + '/' + namepart] = f.resolve()
    for name, path in (extra_files or {}).items():
        if Path(root).resolve() in Path(path).resolve().parents:
            raise ValueError('Case-local dependency files belong in the copied initial tree, not shared fingerprints')
        paths['declared:' + name] = Path(path).resolve()
    files = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise ValueError('Shared runtime fingerprint file missing: ' + str(path))
        files[name] = {'path': str(path), 'sha256': _sha(path), 'mode': stat.S_IMODE(path.stat().st_mode)}
    return {'files': files, 'packages': packages,
            'shared_stores_read_only_under_runner_outer_policy': True,
            'scope': 'Exact listed binaries/source files, R package DESCRIPTIONs (plus selected core R code), Python installed distribution metadata/RECORD. This is not a byte claim for every host file. Case-local packages are copied in the initial tree.'}


def _cold_case(root, runner):
    operator = _read_json(root / 'operator.json')
    allowed = set(TREES) | {'operator.json', 'funded-relay-binding.json', 'trusted-boundary-canary.json'}
    if set(p.name for p in root.iterdir()) - allowed:
        raise ValueError('Case already has analysis/phase/state outputs or an earlier capture; no retrospective initial snapshot')
    entries = _inventory(root)
    workspace = {p[len('workspace/'):]: x['sha256'] for p, x in entries.items()
                 if p.startswith('workspace/') and x['kind'] == 'file'}
    if workspace != operator['packet_sha256']:
        raise ValueError('Workspace differs from original initialization; outputs cannot be backfilled as initial inputs')
    if set(p for p in entries if p.startswith('codex-home/')) != {CONFIG}:
        raise ValueError('Native session/cache/auth state exists; capture must precede analysis')
    if DATA not in entries:
        raise ValueError('Initial data.csv required')
    text = (root / CONFIG).read_text()
    config_kwargs = ({'developer_instructions_extra': operator['developer_instructions_extra']}
                     if _linux(runner) and 'developer_instructions_extra' in operator else {})
    if (operator['runner_sha256'] != _sha(runner.__file__) or
        operator['config_sha256'] != _sha(root / CONFIG) or
        text != runner.config_text(operator['relay_base_url'], root, operator['network_profile'],
                                   **config_kwargs)):
        raise ValueError('Initialized runner/config binding differs')
    for key, path in _source_files(runner).items():
        if operator[key] != _sha(path):
            raise ValueError('Frozen tool/prompt/network source changed')
    if operator['cli_sha256'] != _sha(runner.CLI):
        raise ValueError('Frozen CLI changed')
    return operator, entries


def freeze_initial_state(initialized_case_root, destination, *, runner=None, neutral_parent=None,
                         shared_runtime_files=None, runtime_receipt=None):
    runner = _load_runner(runner)
    linux = _linux(runner)
    parent = neutral_parent or (Path(initialized_case_root).absolute().parent if linux else '/private/tmp')
    root = _opaque_root(initialized_case_root, parent, guest_paths=linux)
    destination = Path(destination).resolve()
    if destination.exists() or destination == root or root in destination.parents:
        raise ValueError('Fresh snapshot outside case root required')
    operator, entries = _cold_case(root, runner)
    runtime_receipt = _linux_receipt(runtime_receipt, runner) if linux else None
    shared = _shared_fingerprints(runner, root, shared_runtime_files, runtime_receipt)
    actual_policy = _policy(runner, root, operator['relay_base_url'], operator)
    destination.mkdir(parents=True, mode=0o700)
    tree = destination / 'tree'; tree.mkdir(mode=0o700)
    for name in TREES:
        shutil.copytree(root / name, tree / name, copy_function=shutil.copy2)
    if _inventory(root) != entries or _inventory(tree) != entries:
        raise ValueError('Initial state changed during capture; incomplete snapshot retained')
    manifest = _write_json(destination / 'manifest.json', {
        'schema': 'native-cli-initial-state-v1', 'original_root': str(root), 'neutral_parent': str(root.parent),
        'root_mode': stat.S_IMODE(root.stat().st_mode), 'operator': operator, 'entries': entries,
        'normalized_config': _normalized_config((root / CONFIG).read_text(), root, operator['relay_base_url'], runner),
        'guest_paths': dict(LINUX_GUEST_PATHS) if linux else None, 'runtime_receipt': runtime_receipt,
        # Current initialize_case hashes packet deny rules in caller insertion
        # order; invoke_offline reconstructs them from sorted operator.json.
        # Bind the actual invocation policy explicitly without rewriting either.
        'invocation_policy': actual_policy,
        'normalized_invocation_policy': _normalized_policy(actual_policy,
            root, operator['relay_base_url'], runner),
        'invocation_policy_sha256': _policy_hash(actual_policy),
        'initializer_policy_sha256': operator['outer_policy_sha256'],
        'shared_runtime': shared, 'shared_runtime_files': {k: str(Path(v).resolve()) for k, v in (shared_runtime_files or {}).items()},
        'capture_boundary': 'verified cold initialized case before native analysis; never reconstructed from an analyzed case',
        'excluded_curator_metadata': sorted(set(p.name for p in root.iterdir()) - set(TREES) - {'operator.json'}),
        'provider_calls': 0})
    # Keep snapshot payload read-only; manifest records the permissions to restore.
    for p in sorted(tree.rglob('*'), reverse=True):
        p.chmod(stat.S_IMODE(p.stat().st_mode) & ~0o222)
    tree.chmod(0o500); (destination / 'manifest.json').chmod(0o400)
    _write_json(root / 'initial-state-frozen.json', {'schema': 'native-cli-initial-capture-receipt-v1',
        'snapshot_hash': manifest['hash'], 'snapshot': str(destination), 'operator_hash': operator['hash']})
    return manifest


def _relocated_source_comparison(current, expected):
    """Compare host-side source relocations only after _linux_fingerprints checks bytes/modes.

    Curator metadata is copied, never rewritten. Guest paths, binaries, packages,
    receipt, source content, source modes and every non-source field stay exact.
    """
    names = {'source:runner_sha256': 'linux_runner.py',
             'source:network_policy_sha256': 'network_policy.py',
             'source:commitment_prompt_sha256': 'commitment_prompt.txt',
             'source:comparison_prompt_sha256': 'comparison_prompt.txt',
             'source:linux_boundary_sha256': 'linux_boundary.py'}
    out = dict(current); out['files'] = dict(current['files'])
    relocations = {}
    for key, filename in names.items():
        a, b = current['files'].get(key), expected['files'].get(key)
        if a is None or b is None:
            raise ValueError('Frozen source relocation key absent: ' + key)
        if a['path'] == b['path']:
            continue
        # This exception covers only the unchanged candidate source bundle's
        # internal locations; it cannot move dependency/library or guest paths.
        suffix = ('candidate', 'studies', 'codex_cli', filename)
        if (tuple(Path(a['path']).parts[-4:]) != suffix or
                tuple(Path(b['path']).parts[-4:]) != suffix or
                {k:v for k,v in a.items() if k != 'path'} !=
                {k:v for k,v in b.items() if k != 'path'}):
            raise ValueError('Frozen source relocation content/mode/layout changed: ' + key)
        relocations[key] = {'original_host_path': b['path'], 'current_host_path': a['path'],
                            'sha256': a['sha256'], 'mode': a['mode']}
        out['files'][key] = dict(a, path=b['path'])
    return out, relocations


def _verify_snapshot(snapshot, runner):
    snapshot = Path(snapshot).resolve(); manifest = _read_json(snapshot / 'manifest.json')
    if manifest.get('schema') != 'native-cli-initial-state-v1':
        raise ValueError('Wrong initial snapshot schema')
    observed = _inventory(snapshot / 'tree')
    expected = manifest['entries']
    for path in expected:
        _safe_relative(path)
    if set(observed) != set(expected):
        raise ValueError('Initial snapshot paths changed')
    for path in expected:
        a, b = dict(observed[path]), dict(expected[path]); a.pop('mode'); b.pop('mode')
        if a != b:
            raise ValueError('Initial snapshot bytes changed: ' + path)
    if manifest.get('guest_paths') != (LINUX_GUEST_PATHS if _linux(runner) else None):
        raise ValueError('Snapshot platform/guest paths differ')
    current_shared = _shared_fingerprints(runner, Path(manifest['original_root']), manifest['shared_runtime_files'],
                                          manifest.get('runtime_receipt'))
    expected_shared = manifest['shared_runtime']
    if _linux(runner):
        # Legacy snapshots derived this list from dictionary insertion order.
        # Normalize only list ordering in comparison copies; never rewrite the
        # signed manifest, dependency bindings, source paths, or receipt hash.
        current_shared = dict(current_shared)
        expected_shared = dict(expected_shared)
        current_shared['read_only_roots'] = sorted(current_shared['read_only_roots'])
        expected_shared['read_only_roots'] = sorted(expected_shared['read_only_roots'])
        current_shared, _ = _relocated_source_comparison(current_shared, expected_shared)
    if current_shared != expected_shared:
        raise ValueError('Shared read-only runtime/package fingerprint changed')
    return manifest


def clone_initial_state(snapshot, new_opaque_case_root, replacement_data_csv, *, runner=None, base_url=None,
                        runtime_receipt=None):
    runner = _load_runner(runner); snapshot = Path(snapshot).resolve()
    manifest = _verify_snapshot(snapshot, runner)
    linux = _linux(runner)
    if runtime_receipt is not None and (not linux or _linux_receipt(runtime_receipt, runner) != manifest['runtime_receipt']):
        raise ValueError('Replay runtime receipt differs from frozen initial runtime')
    parent = Path(new_opaque_case_root).absolute().parent if linux else manifest['neutral_parent']
    root = _opaque_root(new_opaque_case_root, parent, guest_paths=linux)
    if root.exists():
        raise ValueError('Fresh replay root required')
    data = Path(replacement_data_csv)
    if data.is_symlink() or not data.is_file():
        raise ValueError('Replacement data must be a regular file')
    original = manifest['operator']; base_url = original['relay_base_url'] if base_url is None else base_url
    runner.relay_port(base_url)
    packet = {name: (data if name == 'data.csv' else snapshot / 'tree/workspace' / name)
              for name in original['packet_sha256']}
    kwargs = {'network_profile': original['network_profile']}
    if linux: kwargs['runtime_receipt'] = manifest['runtime_receipt']
    if linux and 'developer_instructions_extra' in original:
        kwargs['developer_instructions_extra'] = original['developer_instructions_extra']
    fresh = runner.initialize_case(root, packet, original['initial_prompt'], base_url, original['limits'], **kwargs)
    config_text = (root / CONFIG).read_text()
    if _normalized_config(config_text, root, base_url, runner) != manifest['normalized_config']:
        raise ValueError('Initializer changed technical configuration; clone retained but not verified')
    actual_policy = _policy(runner, root, base_url, fresh)
    if _normalized_policy(actual_policy, root, base_url, runner) != manifest['normalized_invocation_policy']:
        raise ValueError('Clone changed actual permission/network policy')
    allowed_operator_changes = {'hash', 'config_sha256', 'packet_sha256', 'outer_policy_sha256', 'relay_base_url'}
    if {k: v for k, v in original.items() if k not in allowed_operator_changes} != {
            k: v for k, v in fresh.items() if k not in allowed_operator_changes}:
        raise ValueError('Initializer changed frozen model/tool/operator fields')
    old_values = _config_values((snapshot / 'tree' / CONFIG).read_text()); new_values = _config_values(config_text)
    differences = {k: {'before': old_values[k], 'after': new_values[k]} for k in old_values if old_values[k] != new_values[k]}
    if set(differences) - (PATH_KEYS | {RELAY_KEY}):
        raise ValueError('Unapproved config difference')
    if linux and set(differences) - {RELAY_KEY}:
        raise ValueError('Linux case config must be byte-identical apart from an explicit relay change')
    for relative, entry in manifest['entries'].items():
        target = root / relative
        if entry['kind'] == 'directory':
            target.mkdir(parents=True, exist_ok=True)
        elif relative != CONFIG:
            shutil.copyfile(data if relative == DATA else snapshot / 'tree' / relative, target)
        target.chmod(entry['mode'])
    root.chmod(manifest['root_mode'])
    expected = json.loads(json.dumps(manifest['entries']))
    for relative in (DATA, CONFIG):
        expected[relative].update(sha256=_sha(root / relative), size=(root / relative).stat().st_size)
    actual = _inventory(root)
    if actual != expected:
        raise ValueError('Clone paths, contents or modes differ beyond allowed substitution')
    if runner.inventory(root / 'workspace') != fresh['packet_sha256']:
        raise ValueError('Clone packet differs from new operator binding')
    receipt = _write_json(root / 'initial-state-clone.json', {
        'schema': 'native-cli-initial-clone-v1', 'snapshot_hash': manifest['hash'], 'operator_hash': fresh['hash'],
        'entries': actual, 'allowed_data_difference': {'path': DATA, 'before': manifest['entries'][DATA], 'after': actual[DATA]},
        'allowed_config_differences': differences, 'normalized_config_identical': True,
        'allowed_permission_policy_differences': _policy_differences(manifest.get('invocation_policy'), actual_policy),
        'config_bytes_identical': (root / CONFIG).read_bytes() == (snapshot / 'tree' / CONFIG).read_bytes(),
        'normalized_permission_policy_identical': True,
        'shared_runtime_hash': _hash(manifest['shared_runtime']), 'provider_calls': 0})
    return receipt


def verify_clone_initial_state(snapshot, case_root, *, runner=None):
    runner = _load_runner(runner); manifest = _verify_snapshot(snapshot, runner)
    parent = Path(case_root).absolute().parent if _linux(runner) else manifest['neutral_parent']
    root = _opaque_root(case_root, parent, guest_paths=_linux(runner)); receipt = _read_json(root / 'initial-state-clone.json')
    operator = _read_json(root / 'operator.json')
    if (receipt['snapshot_hash'] != manifest['hash'] or receipt['operator_hash'] != operator['hash'] or
            _inventory(root) != receipt['entries'] or
            _normalized_config((root / CONFIG).read_text(), root, operator['relay_base_url'], runner) != manifest['normalized_config'] or
            _normalized_policy(_policy(runner, root, operator['relay_base_url'], operator), root,
                operator['relay_base_url'], runner) != manifest['normalized_invocation_policy']):
        raise ValueError('Clone changed before analysis')
    return receipt
