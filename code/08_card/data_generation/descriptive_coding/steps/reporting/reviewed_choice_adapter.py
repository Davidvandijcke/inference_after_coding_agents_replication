"""Exact reviewed declaration-code adapter. No process features/strata are joined.

The case map is projected to coding_id only. Matching status and strata do not
enter this adapter. Formal declarations retain external artifact provenance;
an analysis-event citation is neither supplied nor fabricated.
"""
import argparse
import csv
import hashlib
import gzip
import json
from pathlib import Path
from features import canonical, digest, check

VERSION = 'card-external-reviewed-declaration-adapter-v1'
EXACT_CODES = {'nonmanager_fte': 'fte', 'manager_inclusive_fte': 'fte',
               'full_time_employees': 'fulltime', 'employee_headcount': 'headcount',
               'other_unclear': None}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl_records(path):
    path = Path(path).resolve()
    artifact = {'path': str(path), 'sha256': file_sha(path)}
    records = []
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as f:
        for line, raw in enumerate(f, 1):
            if raw.strip():
                records.append((json.loads(raw), {**artifact, 'line': line,
                                                  'record_sha256': hashlib.sha256(raw).hexdigest()}))
    return records, artifact


def roster_ids(path):
    path = Path(path).resolve()
    artifact = {'path': str(path), 'sha256': file_sha(path)}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        check('coding_id' in (reader.fieldnames or []), 'case map needs coding_id')
        # All other columns are intentionally ignored, including stratum/match.
        rows = [(r['coding_id'], {**artifact, 'csv_record': rank}) for rank, r in enumerate(reader, 1)]
    return rows, artifact


def build_adapter(declarations, roster, source_index):
    """Pure exact mapping. Inputs include per-record immutable provenance."""
    ids = [identifier for identifier, _ in roster]
    check(len(ids) == len(set(ids)), 'duplicate roster coding IDs')
    declared = {row['coding_id']: (row, provenance) for row, provenance in declarations}
    check(len(declared) == len(declarations), 'duplicate declaration coding IDs')
    check(set(declared) == set(ids), 'declarations must cover all and only replay roster IDs')
    metadata = {row['coding_id']: (row, provenance) for row, provenance in source_index if not row['is_original']}
    check(len(metadata) == sum(not row['is_original'] for row, _ in source_index), 'duplicate source coding IDs')
    check(set(metadata) == set(ids), 'analysis metadata must cover every replay ID exactly once')
    result = {}
    for identifier, roster_provenance in sorted(roster):
        declaration, declaration_provenance = declared[identifier]
        source, source_provenance = metadata[identifier]
        token = declaration['display']['employment_measure']
        check(token in EXACT_CODES, 'unmapped reviewed outcome code; explicit Root mapping required: ' + str(token))
        group = EXACT_CODES[token]
        result[identifier] = {
            'schema': VERSION, 'id': identifier, 'is_original': False,
            'source_sha256': source['eventstream_sha256'],
            'source_kind': 'external_reviewed_declaration',
            'declaration_sha256': declaration['declaration_sha256'],
            'source_artifact': declaration_provenance,
            'review_ref': declaration_provenance['path'] + ':' + str(declaration_provenance['line']),
            'status': 'confirmed' if group is not None else 'unresolved',
            'groups': [group] if group is not None else [],
            'raw_codes': declaration['display'],
            'raw_declaration_record': declaration,
            'case_roster_artifact': roster_provenance,
            'analysis_source_binding': {'packet_sha256': source['packet_sha256'],
                'eventstream_sha256': source['eventstream_sha256'], 'source_index_artifact': source_provenance},
        }
    originals = []
    for source, provenance in source_index:
        if source['is_original']:
            originals.append({'id': source.get('coding_id') or source['case_id'], 'is_original': True,
                'status': 'unresolved', 'groups': [], 'source_sha256': source['eventstream_sha256'],
                'reason': 'Replay formal-declaration codebook supplies no original declaration row; original remains separate.',
                'analysis_source_binding': {'packet_sha256': source['packet_sha256'],
                    'eventstream_sha256': source['eventstream_sha256'], 'source_index_artifact': provenance}})
    return {'schema': VERSION, 'exact_mapping': EXACT_CODES, 'replays_by_coding_id': result,
            'original_separate': sorted(originals, key=lambda r: r['id']),
            'process_features_joined': False, 'strata_joined': False,
            'matching_uncertainty_excluded': False}


def write_adapter(document, destination, inputs, expected_replays=3000):
    check(len(document['replays_by_coding_id']) == expected_replays, 'unexpected replay roster size')
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    names = ('REVIEWED_CHOICE_ADAPTER.json', 'reviewed-choice-adapter.jsonl', 'reviewed-choice-adapter.csv', 'ADAPTER_RECEIPT.json')
    check(not any((dest / n).exists() for n in names), 'adapter outputs are immutable; use a new destination')
    with (dest / names[0]).open('x') as f:
        json.dump(document, f, sort_keys=True, indent=2); f.write('\n')
    with (dest / names[1]).open('xb') as f:
        for row in document['replays_by_coding_id'].values():
            f.write(canonical(row) + b'\n')
    with (dest / names[2]).open('x', newline='') as f:
        fields = ['id', 'source_sha256', 'source_kind', 'declaration_sha256', 'status', 'groups_json',
                  'employment_measure_code', 'source_artifact_json', 'case_roster_artifact_json', 'analysis_source_binding_json']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in document['replays_by_coding_id'].values():
            writer.writerow({**{key: row[key] for key in fields[:5]}, 'groups_json': canonical(row['groups']).decode(),
                'employment_measure_code': row['raw_codes']['employment_measure'],
                **{key + '_json': canonical(row[key]).decode() for key in
                   ('source_artifact', 'case_roster_artifact', 'analysis_source_binding')}})
    receipt = {'schema': VERSION, 'replay_records': expected_replays,
        'original_separate_records': len(document['original_separate']), 'inputs': inputs,
        'mapping_sha256': digest(EXACT_CODES), 'document_sha256': digest(document),
        'outputs': {name: file_sha(dest / name) for name in names[:3]},
        'process_features_joined': False, 'strata_joined': False,
        'matching_status_used_to_filter': False, 'analysis_event_citations_fabricated': False}
    with (dest / names[3]).open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-codes', required=True, type=Path)
    p.add_argument('--case-map', required=True, type=Path)
    p.add_argument('--source-index', required=True, type=Path)
    p.add_argument('--destination', required=True, type=Path)
    args = p.parse_args()
    declarations, final_provenance = jsonl_records(args.final_codes)
    roster, roster_provenance = roster_ids(args.case_map)
    index, index_provenance = jsonl_records(args.source_index)
    document = build_adapter(declarations, roster, index)
    receipt = write_adapter(document, args.destination,
        {'final_codes': final_provenance, 'case_map': roster_provenance, 'source_index': index_provenance})
    print(json.dumps({'replay_records': receipt['replay_records'],
                      'original_separate_records': receipt['original_separate_records'],
                      'document_sha256': receipt['document_sha256'],
                      'process_features_joined': False, 'strata_joined': False}, sort_keys=True))
