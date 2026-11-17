"""Credential-free Linux bubblewrap boundary for native CLI and fixed scorers.

Source-only construction: no network, process, provider, or Slurm activity on
import. A caller must bind an actual trusted runtime receipt and free canary
before any funded invocation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat


GUEST_PATHS = {'workspace': '/workspace', 'home': '/home/researcher',
               'tmp': '/tmp', 'codex-home': '/codex-home'}
GUEST_CLI = '/opt/codex/codex'
GUEST_CODE_MODE_HOST = '/opt/codex/codex-code-mode-host'
GUEST_SCHEMA = '/run/scientific/schema.json'
PROXY_URL = 'http://<cluster-proxy>:3128'
NO_PROXY = 'localhost,127.0.0.1,::1'
FORBIDDEN_GUEST_PREFIXES = ('/home', '/scratch', '/workspace', '/codex-home',
                            '/tmp', '/run', '/opt/codex')
PUBLIC_HOST_PREFIXES = ('/usr', '/bin', '/sbin', '/lib', '/lib64', '/sw', '/opt')
PUBLIC_ETC = {'/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf',
              '/etc/passwd', '/etc/group', '/etc/localtime', '/etc/ssl',
              '/etc/pki', '/etc/ld.so.cache', '/etc/fonts', '/etc/os-release',
              '/etc/services', '/etc/protocols', '/etc/alternatives'}
CURATED_HOST = '<cluster-scratch>/immigration-scientific-python-v1'
CURATED_BUILD = CURATED_HOST + '-build'
CURATED_GUEST = '/opt/scientific-python'
CURATED_RECEIPT_SHA256 = '728e626628f69e5a38b43cdc4c25e4f5bd37cfde6fd301c5033a6ef24ea69f13'


def _curated_receipt(binding):
    if (not isinstance(binding, dict) or
        binding.get('receipt_file') != CURATED_BUILD + '/prefix-receipt.json' or
        binding.get('receipt_sha256') != CURATED_RECEIPT_SHA256 or
        sha_file(binding['receipt_file']) != binding['receipt_sha256']):
        raise ValueError('Pinned curated Python prefix receipt differs')
    receipt = json.loads(Path(binding['receipt_file']).read_text())
    inventory = CURATED_BUILD + '/site-files.json'
    if (receipt.get('schema') != 'public-scientific-python-prefix-v1' or
        receipt.get('complete') is not True or
        str(Path(receipt.get('site_packages', '')).resolve()) != str(Path(CURATED_HOST).resolve()) or
        str(Path(receipt.get('site_inventory', '')).resolve()) != str(Path(inventory).resolve()) or
        receipt.get('guest_mount') != CURATED_GUEST or
        receipt.get('guest_pythonpath') != CURATED_GUEST or
        receipt.get('provider_calls') != 0 or
        receipt.get('scientific_programs_executed') != 0 or
        type(receipt.get('file_count')) is not int or receipt['file_count'] < 1 or
        type(receipt.get('bytes')) is not int or receipt['bytes'] < 1 or
        not isinstance(receipt.get('site_inventory_sha256'), str) or
        sha_file(inventory) != receipt['site_inventory_sha256'] or
        not isinstance(receipt.get('packages'), list) or
        not isinstance(receipt.get('imports'), dict) or
        not isinstance(receipt.get('python'), str) or
        not isinstance(receipt.get('python_version'), str) or
        not isinstance(receipt.get('python_sha256'), str) or
        len(receipt['python_sha256']) != 64 or
        not isinstance(receipt.get('lock_sha256'), str) or
        len(receipt['lock_sha256']) != 64):
        raise ValueError('Curated Python prefix receipt/inventory binding differs')
    return {'host_root': CURATED_HOST, 'guest_root': CURATED_GUEST,
            'receipt_file': binding['receipt_file'],
            'receipt_sha256': binding['receipt_sha256'],
            'site_inventory': inventory,
            'site_inventory_sha256': receipt['site_inventory_sha256'],
            'file_count': receipt['file_count'], 'bytes': receipt['bytes'],
            'python_sha256': receipt.get('python_sha256')}


def verify_curated_prefix_full(runtime):
    binding = runtime.get('curated_python_prefix')
    if binding is None:
        return None
    checked = _curated_receipt(binding)
    if checked != binding:
        raise ValueError('Curated Python prefix normalization differs')
    root = Path(CURATED_HOST)
    if root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise ValueError('Curated prefix root is linked, missing, or writable')
    entries = json.loads(Path(binding['site_inventory']).read_text())
    if not isinstance(entries, list) or len(entries) != binding['file_count']:
        raise ValueError('Curated prefix inventory count differs')
    expected = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {'path', 'sha256', 'bytes', 'mode'}:
            raise ValueError('Malformed curated file inventory entry')
        relative = item['path']
        p = PurePosixPath(relative) if isinstance(relative, str) else None
        if (p is None or p.is_absolute() or not p.parts or '..' in p.parts or
            str(p) != relative or relative in expected or
            type(item['bytes']) is not int or item['bytes'] < 0 or
            item['mode'] not in (0o444, 0o555) or
            not isinstance(item['sha256'], str) or len(item['sha256']) != 64):
            raise ValueError('Unsafe curated file inventory entry')
        expected[relative] = item
    # The receipt pins the inventory's exact order and bytes. Its producer
    # enumerates sorted directories, which need not equal global path sorting.
    observed = set(); byte_count = 0
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            directory = Path(parent) / name
            if directory.is_symlink() or stat.S_IMODE(directory.stat().st_mode) != 0o555:
                raise ValueError('Curated prefix directory is linked or writable')
        for name in files:
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or relative not in expected:
                raise ValueError('Unexpected or linked curated prefix file')
            item = expected[relative]
            if (path.stat().st_size != item['bytes'] or
                stat.S_IMODE(path.stat().st_mode) != item['mode'] or
                sha_file(path) != item['sha256']):
                raise ValueError('Curated prefix file bytes or mode changed')
            observed.add(relative); byte_count += item['bytes']
    if observed != set(expected) or byte_count != binding['bytes']:
        raise ValueError('Curated prefix inventory set or byte total differs')
    return {'receipt_sha256': binding['receipt_sha256'],
            'site_inventory_sha256': binding['site_inventory_sha256'],
            'verified_files': len(observed), 'verified_bytes': byte_count}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def sha_file(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError('Regular runtime file required: ' + str(p))
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _inside(path, prefix):
    return path == prefix or path.startswith(prefix + '/')


def load_runtime_receipt(value):
    from_file = not isinstance(value, dict)
    raw = json.loads(Path(value).read_text()) if from_file else dict(value)
    if raw.get('schema') == 'greatlakes-native-runtime-environment-v1':
        curated = (_curated_receipt(raw['curated_python_prefix'])
                   if raw.get('curated_python_prefix') is not None else None)
        if (curated is not None and
            sha_file(Path(raw['host_python']).resolve()) != curated['python_sha256']):
            raise ValueError('Curated prefix Python interpreter differs from runtime')
        mounts = {guest: host for host, guest in raw['public_readonly_mounts']}
        if curated is not None:
            if CURATED_GUEST in mounts:
                raise ValueError('Curated prefix must not be an ordinary OS mount')
            mounts[CURATED_GUEST] = CURATED_HOST
        x = {'schema': 'scientific-linux-runtime-v1',
             'source_schema': raw['schema'],
             'source_file_sha256': sha_file(value) if from_file else raw.get('source_file_sha256'),
             'host_cli': raw['host_cli'], 'cli_sha256': raw['cli_sha256'],
             'host_code_mode_host': raw['host_code_mode_host'],
             'code_mode_host_sha256': raw['code_mode_host_sha256'],
             'bwrap': raw['bwrap'], 'bwrap_sha256': raw['bwrap_sha256'],
             'public_mounts': mounts,
             'curated_python_prefix': curated,
             'guest_runtime_env': raw['guest_runtime_environment'],
             'network_proxy': raw.get('network_proxy'),
             'environment_job_id': raw['job_id'], 'guest_cli': raw['guest_cli'],
             'guest_python': raw['host_python'],
             'network_profile': raw['network_profile']}
        x['hash'] = digest(x)
    else:
        x = raw
        claim = x.pop('hash', None)
        if x.get('schema') != 'scientific-linux-runtime-v1' or claim != digest(x):
            raise ValueError('Trusted Linux runtime receipt hash/schema differs')
        x['hash'] = claim
        if x.get('curated_python_prefix') is not None:
            if _curated_receipt(x['curated_python_prefix']) != x['curated_python_prefix']:
                raise ValueError('Curated prefix receipt normalization differs')
    for name in ('host_cli', 'cli_sha256', 'host_code_mode_host',
                 'code_mode_host_sha256', 'bwrap', 'bwrap_sha256'):
        if not isinstance(x.get(name), str):
            raise ValueError('Incomplete Linux runtime receipt: ' + name)
    if (sha_file(x['host_cli']) != x['cli_sha256'] or
        sha_file(x['host_code_mode_host']) != x['code_mode_host_sha256'] or
        sha_file(x['bwrap']) != x['bwrap_sha256']):
        raise ValueError('Pinned Codex executable, code-mode host, or bwrap bytes changed')
    mounts = x.get('public_mounts')
    if not isinstance(mounts, dict) or not mounts:
        raise ValueError('Explicit public Linux runtime mounts required')
    for guest, host in mounts.items():
        if not isinstance(guest, str) or not guest.startswith('/') or guest == '/':
            raise ValueError('Public guest mount path invalid')
        if any(_inside(guest, forbidden) for forbidden in FORBIDDEN_GUEST_PREFIXES):
            raise ValueError('Public mount overlaps private guest path')
        p = Path(host)
        if not p.is_absolute() or not p.exists():
            raise ValueError('Public host mount missing or relative')
        real = str(p.resolve())
        if guest == CURATED_GUEST and host == CURATED_HOST:
            if (x.get('curated_python_prefix') is None or
                real != str(Path(CURATED_HOST).resolve())):
                raise ValueError('Unpinned curated Python mount')
            continue
        if not (any(_inside(real, prefix) for prefix in PUBLIC_HOST_PREFIXES) or
                real in PUBLIC_ETC or str(p) in PUBLIC_ETC or
                any(_inside(real, prefix) for prefix in ('/etc/ssl', '/etc/pki'))):
            raise ValueError('Public mount host source is outside OS/module roots')
    guest_env = x.get('guest_runtime_env')
    if not isinstance(guest_env, dict) or not isinstance(guest_env.get('PATH'), str):
        raise ValueError('Explicit module-bound guest PATH required')
    permitted = {'PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH',
                 'PKG_CONFIG_PATH', 'R_LIBS_USER', 'R_LIBS_SITE',
                 'PYTHONPATH', 'LANG', 'LC_ALL', 'OMP_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE',
                 'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                 'NO_PROXY', 'no_proxy', 'PYTHONDONTWRITEBYTECODE'}
    if set(guest_env) - permitted or any(not isinstance(v, str) for v in guest_env.values()):
        raise ValueError('Unsupported guest runtime environment field')
    proxy_names = ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy')
    if any(name in guest_env for name in proxy_names):
        if (any(guest_env.get(name) != PROXY_URL for name in proxy_names) or
            guest_env.get('NO_PROXY') != NO_PROXY or
            guest_env.get('no_proxy') != NO_PROXY or
            x.get('network_proxy') != PROXY_URL):
            raise ValueError('Guest CONNECT proxy or loopback bypass binding differs')
    if x.get('curated_python_prefix') is not None:
        if (guest_env.get('PYTHONPATH') != CURATED_GUEST or
            guest_env.get('PYTHONDONTWRITEBYTECODE') != '1'):
            raise ValueError('Curated Python guest path/cache policy differs')
    elif any(CURATED_GUEST in value for value in guest_env.values()):
        raise ValueError('Unpinned curated Python path in guest environment')
    for name in ('PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH',
                 'PKG_CONFIG_PATH', 'R_LIBS_USER', 'R_LIBS_SITE', 'PYTHONPATH'):
        for item in guest_env.get(name, '').split(':'):
            case_local = _inside(item, GUEST_PATHS['home']) and name in ('PATH', 'R_LIBS_USER', 'PYTHONPATH')
            if item and (not item.startswith('/') or
                         (any(_inside(item, p) for p in
                              ('/home', '/scratch', '/workspace', '/codex-home', '/tmp'))
                          and not case_local)):
                raise ValueError('Guest runtime path points outside public mounts')
    return x


def _mkdir_flags(paths):
    result, seen = [], {'/'}
    for target in paths:
        p = Path(target)
        for parent in reversed(p.parents):
            if str(parent) not in seen:
                seen.add(str(parent)); result += ['--dir', str(parent)]
        if str(p) not in seen:
            seen.add(str(p)); result += ['--dir', str(p)]
    return result


def initial_permission_policy(case_root, base_url, network_profile, packet_names, runtime_receipt):
    from urllib.parse import urlparse
    u = urlparse(base_url)
    if (u.scheme != 'http' or u.hostname != '127.0.0.1' or not u.port or
            u.path != '/v1' or u.username or u.password or u.query or u.fragment):
        raise ValueError('Explicit host-loopback billing relay required')
    if network_profile != 'public_internet':
        raise ValueError('Linux shared-network boundary is approved only for public_internet pilots')
    runtime = load_runtime_receipt(runtime_receipt)
    root = Path(case_root).resolve()
    names = sorted(packet_names)
    if any(not name or name != Path(name).name or name.startswith('.') or
           name.startswith('AGENTS') for name in names):
        raise ValueError('Only approved top-level packet names may be mounted')
    return {'schema': 'scientific-linux-bwrap-policy-v1',
            'guest_paths': GUEST_PATHS, 'guest_cli': GUEST_CLI,
            'network': 'host_shared_public_internet', 'relay_base_url': base_url,
            'host_case_root': str(root), 'case_binds': {k: str(root / k) for k in GUEST_PATHS},
            'readonly_packet_files': {name: str(root / 'workspace' / name) for name in names},
            'readonly_config': str(root / 'codex-home/config.toml'),
            'public_mounts': runtime['public_mounts'],
            'host_cli': runtime['host_cli'], 'cli_sha256': runtime['cli_sha256'],
            'host_code_mode_host': runtime['host_code_mode_host'],
            'code_mode_host_sha256': runtime['code_mode_host_sha256'],
            'bwrap': runtime['bwrap'], 'bwrap_sha256': runtime['bwrap_sha256'],
            'runtime_receipt_hash': runtime['hash'],
            'guest_runtime_env': runtime['guest_runtime_env'],
            'proc': 'private_pid_namespace_proc_required_by_native_cli_scoped_dummy_token_may_be_visible_to_guest_tools'}


def outer_command(policy, guest_command, *, relay_token=None, schema_file=None,
                  extra_readonly=None, extra_writable=None):
    """Build a bwrap argv; generic R/Python scoring callers omit relay_token.

    Additional binds are exact files/directories with explicit guest locations.
    They are never a whole host home/scratch mount. The returned argv alone is
    not permission to dispatch; the caller must apply funded/health gates.
    """
    if policy.get('schema') != 'scientific-linux-bwrap-policy-v1':
        raise ValueError('Bound Linux policy required')
    # bubblewrap 0.4.0 has no --clearenv. Clear the HOST environment before it
    # starts, then populate only the explicit guest variables with --setenv.
    args = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', policy['bwrap'],
            '--die-with-parent', '--unshare-user', '--unshare-pid',
            '--unshare-ipc', '--new-session', '--tmpfs', '/']
    targets = [guest if Path(host).is_dir() else str(Path(guest).parent)
               for guest, host in policy['public_mounts'].items()]
    targets += list(policy['guest_paths'].values()) + [
        '/dev', '/proc', '/run', '/opt/codex']
    if schema_file is not None:
        targets.append(str(Path(GUEST_SCHEMA).parent))
    targets.extend(str(Path(p).parent) for p in (extra_readonly or {}))
    targets.extend(extra_writable or {})
    args += _mkdir_flags(targets)
    args += ['--dev', '/dev', '--proc', '/proc']
    for guest, host in sorted(policy['public_mounts'].items()):
        args += ['--ro-bind', str(host), guest]
    for name, guest in policy['guest_paths'].items():
        args += ['--bind', policy['case_binds'][name], guest]
    args += ['--ro-bind', policy['host_cli'], GUEST_CLI]
    args += ['--ro-bind', policy['host_code_mode_host'], GUEST_CODE_MODE_HOST]
    for name, host in sorted(policy['readonly_packet_files'].items()):
        args += ['--ro-bind', host, GUEST_PATHS['workspace'] + '/' + name]
    args += ['--ro-bind', policy['readonly_config'], GUEST_PATHS['codex-home'] + '/config.toml']
    if schema_file is not None:
        args += ['--ro-bind', str(Path(schema_file).resolve()), GUEST_SCHEMA]
    for guest, host in sorted((extra_readonly or {}).items()):
        args += ['--ro-bind', str(Path(host).resolve()), guest]
    for guest, host in sorted((extra_writable or {}).items()):
        args += ['--bind', str(Path(host).resolve()), guest]
    environment = dict(policy['guest_runtime_env'])
    environment.update(HOME=GUEST_PATHS['home'], CODEX_HOME=GUEST_PATHS['codex-home'],
                       TMPDIR=GUEST_PATHS['tmp'], TMPPREFIX=GUEST_PATHS['tmp'] + '/zsh')
    if relay_token is not None:
        environment['CODEX_RELAY_TOKEN'] = relay_token
    for key, value in sorted(environment.items()):
        args += ['--setenv', key, value]
    args += ['--chdir', GUEST_PATHS['workspace'], '--'] + list(guest_command)
    return args
