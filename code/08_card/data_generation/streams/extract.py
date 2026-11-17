#!/usr/bin/env python3
"""Portable, label-blind extraction of saved Card analysis streams.

Only recorded closed pack ranges and the recorded original path are read.
Archived commands, code, and output are data: none are executed. Extraction
logic follows card-tool-trace-inventory-20261018-v1/inventory.py/read_trace.py.
No third-party packages, network, databases, or model calls are used.
"""
import argparse
import collections
import gzip
import hashlib
import io
import json
import pathlib
import tarfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INVENTORY = ROOT / 'output/card-tool-trace-inventory-20261018-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def js(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


class Writer:
    """One global gzip member per artifact, deterministic header and JSON."""
    def __init__(self, path):
        self.path = path
        self.file = path.open('wb')
        self.gz = gzip.GzipFile(filename='', fileobj=self.file, mode='wb', compresslevel=9, mtime=0)
        self.rows = 0
        self.raw_bytes = 0

    def write(self, value):
        data = (js(value) + '\n').encode()
        self.gz.write(data)
        self.rows += 1
        self.raw_bytes += len(data)

    def close(self):
        self.gz.close()
        self.file.close()
        return {'path': self.path.name, 'rows': self.rows, 'uncompressed_bytes': self.raw_bytes,
                'bytes': self.path.stat().st_size, 'sha256': sha(self.path.read_bytes())}


def normalize(data, row):
    """Deduplicate items by thread + ID, while preserving all occurrence metadata.

    Raw streams retain full event occurrences verbatim. Canonical item fields are
    the final completion's fields, or the last occurrence if there is no completion.
    First occurrence determines item and unique-command order. An absent item ID
    receives a synthetic, explicitly flagged per-line ID; no timestamp is invented.
    """
    raw_lines = data.splitlines()
    assert all(line.strip() for line in raw_lines), 'blank lines require explicit source-line handling'
    events = [json.loads(line) for line in raw_lines]
    stream_sha256 = sha(data)
    items = {}
    thread = None
    for line, event in enumerate(events, 1):
        if event.get('type') == 'thread.started':
            thread = event.get('thread_id')
        item = event.get('item')
        if not isinstance(item, dict):
            continue
        missing_id = 'id' not in item
        item_id = item.get('id', f'__missing_id_event_line_{line}')
        key = (thread, item_id)
        record = items.setdefault(key, {
            'schema': 'card-neutral-analysis-item-v1', 'case_id': row['case_id'],
            'coding_id': row.get('coding_id'), 'is_original': row['case_id'] == 'original-001',
            'eventstream_sha256': stream_sha256, 'thread_id': thread, 'item_id': item_id,
            'item_id_synthetic': missing_id, 'item_type': item.get('type'),
            'first_event_line': line, 'occurrences': [], 'completed_event_lines': [],
            'started_event_lines': [], 'item_sequence_index': len(items) + 1,
        })
        assert record['item_type'] == item.get('type'), 'item type changed within same thread/ID'
        record['occurrences'].append({'event_line': line, 'event_type': event.get('type'),
                                      'raw_event_sha256': sha(raw_lines[line-1])})
        if event.get('type') == 'item.started':
            record['started_event_lines'].append(line)
        if event.get('type') == 'item.completed':
            record['completed_event_lines'].append(line)
        if event.get('type') == 'item.completed' or not record['completed_event_lines']:
            record['item'] = item
            record['canonical_event_line'] = line
    commands = [record for record in items.values() if record['item_type'] == 'command_execution']
    n = len(commands)
    for index, record in enumerate(commands, 1):
        record.update(command_sequence_index=index,
                      command_sequence_fraction=(index - .5) / n,
                      sequence_third=('early', 'middle', 'late')[min(2, 3 * (index - 1) // n)],
                      command_sha256=sha(record['item'].get('command', '').encode()),
                      output_sha256=sha(record['item'].get('aggregated_output', '').encode()))
    for record in items.values():
        record['last_event_line'] = record['occurrences'][-1]['event_line']
        record['has_completion'] = bool(record['completed_event_lines'])
        record['unmatched_start'] = bool(record['started_event_lines']) and not record['has_completion']
        record['duplicate_completion_count'] = max(0, len(record['completed_event_lines']) - 1)
        # Alignment is an event position, never elapsed or wall-clock time.
        line = record['canonical_event_line']
        preceding = [c for c in commands if c['first_event_line'] < line]
        succeeding = [c for c in commands if c['first_event_line'] > line]
        record['event_alignment'] = {
            'event_line': line, 'event_line_fraction': (line - .5) / len(events),
            'unique_commands_started_before_event': len(preceding),
            'previous_command_index': preceding[-1]['command_sequence_index'] if preceding else None,
            'next_command_index': succeeding[0]['command_sequence_index'] if succeeding else None,
            'position_is_clock_time': False,
        }
    return list(items.values()), len(events)


def extract(destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (INVENTORY / 'inventory.jsonl').open()]
    assert len(rows) == 3001
    assert len({row['case_id'] for row in rows}) == len(rows)
    assert sum(row['case_id'] == 'original-001' for row in rows) == 1
    rows.sort(key=lambda row: (row['case_id'] == 'original-001', row['case_id']))
    raw = Writer(destination / 'analysis_streams.jsonl.gz')
    normalized = Writer(destination / 'analysis_items.jsonl.gz')
    provenance = Writer(destination / 'case_provenance.jsonl.gz')
    cases = Writer(destination / 'neutral_cases.jsonl.gz')
    packs = {}
    totals = collections.Counter()
    item_counts = collections.Counter()
    for number, row in enumerate(rows, 1):
        is_original = row['case_id'] == 'original-001'
        p = {'schema': 'card-analysis-source-v1', 'case_id': row['case_id'],
             'coding_id': row.get('coding_id'), 'is_original': is_original,
             'eventstream_sha256': row['eventstream_sha256'],
             'source_analysis_member': row.get('eventstream_member'),
             'whole_pack_sha256': None, 'whole_pack_sha256_available': False}
        if is_original:
            source = pathlib.Path(row['source_path'])
            data = source.read_bytes()
            p['source_original_path'] = str(source)
            p['source_analysis_member'] = None
        else:
            source = row['source_pack']
            record = source['record']
            path = source['path']
            if path not in packs:
                receipt_bytes = pathlib.Path(source['export_receipt']['path']).read_bytes()
                index_bytes = pathlib.Path(source['index']['path']).read_bytes()
                receipt = json.loads(receipt_bytes)
                assert sha(receipt_bytes) == source['export_receipt']['sha256']
                assert sha(index_bytes) == source['index']['sha256'] == receipt['index_sha256']
                allowed = {(r['case_id'], r['offset'], r['length'], r['payload_sha256'], r['sha256'])
                           for r in map(json.loads, index_bytes.splitlines()) if r.get('kind') == 'case'}
                stat = pathlib.Path(path).stat()
                packs[path] = {'handle': open(path, 'rb'), 'allowed': allowed,
                               'committed': receipt['committed_pack_bytes'],
                               'stat': (stat.st_size, stat.st_mtime_ns)}
            pack = packs[path]
            assert record['kind'] == 'case' and record['case_id'].endswith('-closed')
            assert (record['case_id'], record['offset'], record['length'], record['payload_sha256'],
                    record['index_record_sha256']) in pack['allowed']
            assert record['offset'] + record['length'] <= pack['committed'] <= pack['stat'][0]
            pack['handle'].seek(record['offset'])
            payload = pack['handle'].read(record['length'])
            assert len(payload) == record['length'] and sha(payload) == record['payload_sha256']
            with tarfile.open(fileobj=io.BytesIO(payload), mode='r:') as archive:
                data = archive.extractfile('case/analysis/events.jsonl').read()
                phase = json.loads(archive.extractfile('case/analysis/phase.json').read())
                assert sha(data) == phase['events_sha256']
            p['source_pack'] = source
            p['payload_sha256_verified'] = True
            totals['closed_payload_bytes_read'] += len(payload)
        assert sha(data) == row['eventstream_sha256']
        assert len(data) == row['eventstream_bytes']
        items, lines = normalize(data, row)
        assert lines == row['event_lines']
        commands = [r for r in items if r['item_type'] == 'command_execution']
        assert len(commands) == row['unique_command_count']
        assert sum(r['has_completion'] for r in commands) == row['unique_completed_commands']
        assert sum(r['unmatched_start'] for r in commands) == row['started_unfinished_commands']
        raw.write({'schema': 'card-raw-analysis-stream-v1', 'case_id': row['case_id'],
                   'coding_id': row.get('coding_id'), 'is_original': is_original,
                   'eventstream_sha256': sha(data), 'eventstream_bytes': len(data),
                   'analysis_events_jsonl': data.decode('utf-8')})
        provenance.write(p)
        for item in items:
            normalized.write(item)
            item_counts[('original_' if is_original else 'replay_') + item['item_type']] += 1
        cases.write({'schema': 'card-neutral-case-v1', 'case_id': row['case_id'],
                     'coding_id': row.get('coding_id'), 'is_original': is_original,
                     'eventstream_sha256': sha(data), 'event_lines': lines,
                     'eventstream_bytes': len(data), 'unique_items': len(items),
                     'unique_commands': len(commands),
                     'unmatched_command_starts': sum(r['unmatched_start'] for r in commands),
                     'analysis_phase_returncode': row.get('analysis_phase_returncode'),
                     'analysis_phase_timed_out': row.get('analysis_phase_timed_out')})
        totals['originals' if is_original else 'replays'] += 1
        totals['original_event_lines' if is_original else 'replay_event_lines'] += lines
        totals['source_eventstream_bytes'] += len(data)
        if number % 500 == 0:
            print(js({'extracted_cases': number}), flush=True)
    for path, pack in packs.items():
        stat = pathlib.Path(path).stat()
        assert pack['stat'] == (stat.st_size, stat.st_mtime_ns), 'source pack changed during extraction'
        pack['handle'].close()
    files = [writer.close() for writer in (raw, normalized, provenance, cases)]
    manifest = {
        'schema': 'card-tool-process-portable-extraction-v1',
        'extract_script_sha256': sha(pathlib.Path(__file__).read_bytes()),
        'source_inventory': {'path': str(INVENTORY / 'inventory.jsonl'),
                             'sha256': sha((INVENTORY / 'inventory.jsonl').read_bytes())},
        'source_helpers': [{'path': str(INVENTORY / name), 'sha256': sha((INVENTORY / name).read_bytes())}
                           for name in ('inventory.py', 'read_trace.py', 'SUMMARY.json', 'PILOT_SELECTION.json')],
        'totals': dict(totals), 'unique_item_counts': dict(item_counts), 'files': files,
        'order': 'replay case_id lexical, then original-001; within case first item occurrence',
        'compression': 'one global gzip member per file; level 9; mtime 0; empty filename',
        'deduplication': '(case_id,thread_id,item_id); final completion canonical; all occurrences anchored in raw stream',
        'neutrality': 'No stratum, accepted match label, commitment/comparison body, or numeric correction used or exported. No source code executed.',
        'file_edit_limit': 'File-change events preserve paths/kinds/status. Diffs/content exist only when emitted in command/message text; no unseen file content is reconstructed.',
        'time_limit': 'No timestamps invented. Command thirds and message positions are event/command order only.',
    }
    dump(destination / 'EXTRACTION_MANIFEST.json', manifest)
    print(js({'totals': manifest['totals'], 'files': files}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, default=HERE / 'data')
    args = parser.parse_args()
    extract(args.output.resolve())
