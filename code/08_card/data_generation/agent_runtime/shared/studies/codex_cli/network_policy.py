"""Prospective network policy. No listener, provider call or global setting change."""
import json
from pathlib import Path

PROFILES = ('relay_only', 'public_internet')
DATA_SCOPE = ('Use the supplied packet as the sole research data source. Internet access may be used for technical documentation, '
              'software and package installation. Do not replace or augment research observations with downloaded datasets. '
              'Keep supplied packet files unchanged and write analysis artifacts separately.')


def checked(profile):
    if profile == 'technical_docs_packages':
        raise ValueError('Journal docs/packages profile requires its separate allowlist proxy; unrestricted Internet is not result-blind')
    if profile not in PROFILES:
        raise ValueError('Unknown prospective network profile')
    return profile


def additions(case_root, profile, packet_names):
    checked(profile)
    root = Path(case_root).resolve()
    # Demonstrated R-shell friction: writing /dev/null must be permitted.
    rules = ['(allow file-write* (literal "/dev/null"))']
    for name in packet_names:
        if Path(name).name != name or name.startswith('.'):
            raise ValueError('Invalid immutable public-packet name')
        rules.append(f'(deny file-write* (literal {json.dumps(str(root / "workspace" / name))}))')
    if profile == 'public_internet':
        # Public web ports only. Local services remain inaccessible, except the
        # single relay port explicitly admitted by the base policy.
        for port in (80,443):
            rules.append(f'(allow network-outbound (require-all (remote ip "*:{port}") (require-not (remote ip "localhost:*"))))')
        rules += [
            '(allow network-outbound (remote unix-socket (literal "/private/var/run/mDNSResponder")) (remote ip "*:53"))',
            '(allow mach-lookup (global-name "com.apple.SystemConfiguration.DNSConfiguration") (global-name "com.apple.SystemConfiguration.configd"))',
            '(allow mach-lookup (global-name "com.apple.mDNSResponder") (global-name "com.apple.system.opendirectoryd.libinfo") (global-name "com.apple.trustd") (global-name "com.apple.trustd.agent"))',
            '(allow file-read* (literal "/private/etc/ssl/cert.pem") (literal "/private/etc/resolv.conf") (literal "/private/var/run/resolv.conf"))']
    return '\n'.join(rules)+'\n'
