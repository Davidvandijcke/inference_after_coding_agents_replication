#!/usr/bin/env python3
"""Inert-source, standard-library tooling for Card semantic annotation v3.

No commands from source are executed. No network/API/runtime/ledger operations.
Semantic labels are always supplied by a person or reviewing model.
"""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import itertools
import pathlib
import re
import shlex
import importlib.util
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parent.parent
DEFAULT_RAW = REPO / 'output/card-tool-process-analysis-20261018-v1/engineering/data/analysis_streams.jsonl.gz'
DIMS = ('O', 'S', 'M', 'W', 'T', 'I')
PURPOSE = 'setup inspect construct evaluate interpret choose revise verify debug report other unknown'.split()
EXEC = 'none definition_only evaluated partial_results attempt_failed rerun verification_only mixed unknown'.split()
FEEDBACK = 'new_family_after_result new_family_same_batch revisit_after_result no unknown'.split()
CRITERIA = 'design measurement direction magnitude precision p_significance inference_convention technical other unclear'.split()
ACTIONS = 'next_analysis primary_adopt primary_reject primary_revise report_emphasis none'.split()
KINDS = 'criterion_to_next criterion_to_primary criterion_to_emphasis primary_revision statement_implementation_conflict'.split()
ORDERS = 'before after same_batch_unordered unknown'.split()
STATES = ['prepare/inspect', 'evaluate/surface', 'interpret', 'choose/revise', 'verify', 'report', 'mixed/unresolved']
FAMILIES = {
    'O': 'fte_change fte_level fulltime_change fulltime_level headcount_change headcount_level parttime manager transformed_employment other'.split(),
    'S': 'closure_inclusive respondents_only complete_case status_restriction baseline_restriction chain_or_ownership_subgroup other'.split(),
    'M': 'unadjusted chain_adjusted baseline_outcome_adjusted other_covariate_adjusted interactions_or_heterogeneity other chain+baseline'.split(),
    'W': 'unweighted employment_weighted other_weighted'.split(),
    'T': 'change_contrast level_contrast did_interaction other'.split(),
    'I': 'welch pooled_t classical_ols heteroskedastic_robust cluster_robust bootstrap permutation other'.split(),
}
M_COMPONENTS = ('chain', 'baseline', 'ownership', 'geography', 'other')
M_COMBINATIONS = ['+'.join(x) for n in range(1, 6) for x in itertools.combinations(M_COMPONENTS, n)]
FAMILIES['M'] = list(dict.fromkeys(FAMILIES['M'] + M_COMBINATIONS))
PRODUCTION_FAMILIES = copy.deepcopy(FAMILIES)
PRODUCTION_FAMILIES['M'] = ['unadjusted'] + M_COMBINATIONS + ['interactions_or_heterogeneity']


def family_allowed(dim, family, source):
    if dim not in DIMS:
        return False
    if family in source.vocab[dim]:
        return True
    # Grammar validation only; no source-code interpretation or relabeling.
    parts = family.split('+')
    return not source.production and dim == 'M' and len(parts) == len(set(parts)) and set(parts) <= set(M_COMPONENTS)


GROUPS = {'command_execution': 'commands', 'agent_message': 'messages', 'file_change': 'files'}
SHARED = re.compile(r'\[SHARED OUTPUT ([0-9a-f]{64}); full text in COMMON_OUTPUTS.json\]')
# The sole regex parses an explicit storage reference; it never classifies content.


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def sha(x):
    return hashlib.sha256(x).hexdigest()


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def emit(x):
    print(json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2))


def immutable(path, obj):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: existing artifacts are never rewritten, even identically.
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def strings(value, label):
    require(isinstance(value, list) and all(isinstance(v, str) for v in value), label + ': expected string array')
    require(len(set(value)) == len(value), label + ': duplicate values')


def enums(value, allowed, label):
    strings(value, label)
    require(set(value) <= set(allowed), label + ': invalid enum ' + str(set(value) - set(allowed)))


def integer(v):
    return type(v) is int


def text_fields(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from text_fields(v)
    elif isinstance(value, list):
        for v in value:
            yield from text_fields(v)


class Sources:
    def __init__(self, packets=None, raw=None, vocabulary=None):
        self.directory = pathlib.Path(packets or BASE / 'packets')
        self.raw = pathlib.Path(raw or DEFAULT_RAW)
        self.index = [json.loads(l) for l in (self.directory / 'INDEX.jsonl').read_text().splitlines() if l.strip()]
        self.by_id = {}
        for row in self.index:
            for key in (row['case_id'], row.get('coding_id')):
                if key:
                    require(key not in self.by_id, 'duplicate source identifier: ' + key)
                    self.by_id[key] = row
        self.vocab = copy.deepcopy(FAMILIES)
        self.extra_enums = {}
        self.production = False
        if vocabulary:
            v = load(vocabulary)
            for d, values in v.get('families', {}).items():
                require(d in DIMS, 'unknown vocabulary dimension')
                strings(values, 'vocabulary.' + d)
                self.vocab[d] = values
            self.extra_enums = v.get('enums', {})
        self._packets = {}
        self._raw = {}
        self._verified = set()
        self._rosters = {}

    def collect(self, identifiers):
        wanted = set()
        for i in identifiers:
            require(i in self.by_id, 'unknown case: ' + str(i))
            wanted.add(self.by_id[i]['case_id'])
        need = wanted - self._packets.keys()
        if need:
            with gzip.open(self.directory / 'neutral-packets.jsonl.gz', 'rt') as f:
                for line in f:
                    p = json.loads(line)
                    if p['case_id'] in need:
                        ix = self.by_id[p['case_id']]
                        require(p['eventstream_sha256'] == ix['eventstream_sha256'], 'packet/index source hash mismatch')
                        # Root's packet hash binds the exact UTF-8 JSONL record including newline.
                        require(sha(line.encode()) == ix['packet_sha256'], 'packet record hash mismatch')
                        self._packets[p['case_id']] = p
        require(wanted <= self._packets.keys(), 'packet missing')
        need = wanted - self._raw.keys()
        if need:
            with gzip.open(self.raw, 'rt') as f:
                for line in f:
                    r = json.loads(line)
                    if r['case_id'] in need:
                        data = r['analysis_events_jsonl'].encode()
                        require(sha(data) == r['eventstream_sha256'] == self.by_id[r['case_id']]['eventstream_sha256'], 'raw source hash mismatch')
                        lines = data.decode().splitlines()
                        self._raw[r['case_id']] = (r, lines, [json.loads(x) for x in lines])
        require(wanted <= self._raw.keys(), 'raw stream missing')
        for i in wanted - self._verified:
            self.check_roster(i)
            self._verified.add(i)

    def get(self, identifier):
        self.collect([identifier])
        i = self.by_id[identifier]['case_id']
        return self._packets[i], self._raw[i]

    def check_roster(self, case_id):
        """Match the original extract.normalize callback rule, without source edits.

        Packet final_line is canonical payload line, including an unmatched start.
        Review command at[1] is completion_line, which is null if none exists.
        First observed callback establishes archive order; it is not a clock start.
        """
        p = self._packets[case_id]
        events = self._raw[case_id][2]
        expected = {}
        thread = None
        for n, event in enumerate(events, 1):
            if event.get('type') == 'thread.started':
                thread = event.get('thread_id')
            item = event.get('item', {})
            if item.get('type') not in GROUPS:
                continue
            key = item.get('id', '__missing_id_event_line_' + str(n))
            if key not in expected:
                expected[key] = {'item_id': key, 'type': item['type'], 'thread_id': thread,
                                 'first_line': n, 'first_observed_line': n, 'canonical_event_line': n,
                                 'final_line': None, 'completion_line': None,
                                 'occurrence_lines': [], 'started_lines': [], 'completed_lines': []}
            meta = expected[key]
            require(meta['thread_id'] == thread and meta['type'] == item['type'], 'ambiguous thread/item source identity: ' + key)
            meta['occurrence_lines'].append(n)
            if event.get('type') == 'item.started':
                meta['started_lines'].append(n)
            if event.get('type') == 'item.completed':
                meta['completed_lines'].append(n)
                meta['completion_line'] = n
                meta['final_line'] = n
            if event.get('type') == 'item.completed' or not meta['completed_lines']:
                meta['canonical_event_line'] = n
        actual = {x['item_id']: x for x in p['items']}
        require(len(actual) == len(p['items']), 'duplicate packet item')
        require(set(expected) == set(actual), 'packet/raw roster mismatch')
        for key, item in actual.items():
            meta = expected[key]
            require(item['type'] == meta['type'] and item['first_line'] == meta['first_observed_line'], 'packet/raw roster mismatch: ' + key + '.first/type')
            require(item['final_line'] == meta['canonical_event_line'], 'packet/raw roster mismatch: ' + key + '.canonical_payload_line')
            meta['packet_final_line'] = item['final_line']
            meta['actual_started_line'] = meta['started_lines'][0] if meta['started_lines'] else None
            meta['out_of_order_started_lines'] = [n for n in meta['started_lines'] if meta['completed_lines'] and n > meta['completed_lines'][0]]
            meta['unmatched_start'] = bool(meta['started_lines']) and not meta['completed_lines']
            meta['duplicate_completion_count'] = max(0, len(meta['completed_lines']) - 1)
            meta['command_index'] = item.get('command_index')
        self._rosters[case_id] = expected

    def item_roster(self, identifier):
        self.collect([identifier])
        return self._rosters[self.by_id[identifier]['case_id']]

    def evidence(self, identifier, value, lines_key='ev', require_nonempty=True, check_quote=True):
        p, (_, raw_lines, events) = self.get(identifier)
        lines = value.get(lines_key)
        require(isinstance(lines, list) and all(integer(x) and 1 <= x <= len(events) for x in lines), lines_key + ': invalid source line')
        require(len(lines) == len(set(lines)), lines_key + ': repeated source line')
        if require_nonempty:
            require(bool(lines), lines_key + ': source evidence required')
        if check_quote and 'quote' in value:
            q = value['quote']
            corpus = []
            for n in lines:
                corpus.extend(({'source_line': n, 'source_representation': 'original'}, text) for text in text_fields(events[n - 1]))
                command = events[n - 1].get('item', {}).get('command')
                if isinstance(command, str):
                    decoded = decoded_shell(command)
                    if decoded is not None:
                        corpus.append(({'source_line': n, 'source_representation': 'shell_decoded'}, decoded))
            try:
                return quote_match(q, corpus)
            except Invalid as error:
                raise Invalid('quote absent from cited source events ' + str(lines) + ': ' + str(error)) from error
        return {'source_lines': lines}

    def full_text(self, identifier, raw_events=False):
        p, (_, lines, _) = self.get(identifier)
        common = getattr(self, '_common', None)
        if common is None:
            common = load(self.directory / 'COMMON_OUTPUTS.json')
        def expand(match):
            key = match.group(1)
            require(key in common, 'missing common output: ' + key)
            text = common[key]['text']
            require(sha(text.encode()) == key, 'common output hash mismatch')
            return text
        text = SHARED.sub(expand, p['text'])
        require(not SHARED.search(text), 'unexpanded shared output')
        i = p.get('coding_id') or p['case_id']
        out = [f'FULLSTART {i} source_sha256={p["eventstream_sha256"]}',
               'POSITION_NOTE: packet starts means first observed callback; packet final means canonical payload event. Completion wins over later start callbacks. Positions are archive order, not clock time.', text]
        if raw_events:
            out.append('RAW_SOURCE_EVENTS (inert JSON; original one-based lines)')
            out.extend(f'L{n} {line}' for n, line in enumerate(lines, 1))
        else:
            out.append('SOURCE_METADATA_EVENTS (all item evidence is fully printed above)')
            for n, line in enumerate(lines, 1):
                event = json.loads(line)
                if event.get('item', {}).get('type') not in GROUPS:
                    out.append(f'L{n} {line}')
        out.append(f'FULLEND {i} source_sha256={p["eventstream_sha256"]} lines={len(lines)} items={len(p["items"])}')
        return '\n'.join(out) + '\n'


def quote_match(quote, corpus):
    """Match within one contiguous source field; normalize whitespace only.

    corpus entries are (source_descriptor, original_text). Fields are never joined.
    Punctuation and numeric characters are never rewritten. The original quote,
    matched original source span, and matching method remain in the receipt.
    """
    require(isinstance(quote, str) and bool(quote.strip()), 'quote must be a nonempty source fragment')
    require(len(quote.split()) <= 25, 'quote exceeds 25 words')
    for descriptor, text in corpus:
        if quote in text:
            return dict(descriptor, raw_quote=quote, match_method='verbatim', source_fragment=quote)
    normalized_quote = ' '.join(quote.split())
    for descriptor, text in corpus:
        normalized = ' '.join(text.split())
        pos = normalized.find(normalized_quote)
        if pos < 0:
            continue
        target_end = pos + len(normalized_quote)
        norm_pos, raw_cursor, start, end = 0, 0, None, None
        for word in text.split():
            raw_start = text.find(word, raw_cursor)
            raw_cursor = raw_start + len(word)
            word_end = norm_pos + len(word)
            if start is None and norm_pos <= pos < word_end:
                start = raw_start + pos - norm_pos
            if norm_pos < target_end <= word_end:
                end = raw_start + target_end - norm_pos
                break
            norm_pos = word_end + 1
        require(start is not None and end is not None, 'whitespace match span mapping failed')
        return dict(descriptor, raw_quote=quote, match_method='whitespace_normalized', source_fragment=text[start:end])
    raise Invalid('quote absent from source fields: ' + repr(quote[:160]))


def decoded_shell(command):
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if len(parts) == 3 and pathlib.PurePosixPath(parts[0]).name in ('bash', 'sh', 'zsh') and parts[1] in ('-lc', '-c'):
        return parts[2]
    return None


def item_quote_corpus(record, row, source, group):
    p, (_, _, events) = source.get(record['id'])
    item = source.item_roster(record['id'])[row['id']]
    corpus = []
    for n in item['occurrence_lines']:
        raw_item = events[n - 1].get('item', {})
        fields = ('command', 'aggregated_output', 'output') if group == 'commands' else (('text',) if group == 'messages' else ('changes',))
        for field in fields:
            value = raw_item.get(field)
            values = list(text_fields(value)) if group == 'files' else ([value] if isinstance(value, str) else [])
            for text in values:
                descriptor = {'source_line': n, 'source_field': field, 'source_representation': 'original', 'item_id': row['id']}
                corpus.append((descriptor, text))
                if field == 'command':
                    decoded = decoded_shell(text)
                    if decoded is not None:
                        corpus.append((dict(descriptor, source_representation='shell_decoded'), decoded))
    return corpus


def item_quote_check(record, row, source, group):
    """Bind nonempty <=25-word fragments to their own item, tolerating whitespace."""
    corpus = item_quote_corpus(record, row, source, group)
    q = row.get('quote')
    require(isinstance(q, str) and bool(q.strip()), row['id'] + ': nonempty own-item quote required')
    try:
        return quote_match(q, corpus)
    except Invalid as error:
        raise Invalid(row['id'] + ': quote absent from own original/decoded command, output, or message; ' + str(error)) from error


def coverage_check(record, source):
    """Collect coverage/evidence defects independently of semantic enum validation."""
    p, (_, lines, _) = source.get(record['id'])
    errors = []
    if record.get('hash') != p['eventstream_sha256']:
        errors.append({'path': 'hash', 'error': 'source hash mismatch'})
    groups = {}
    for group in ('commands', 'messages', 'files'):
        roster = {key: i for key, i in source.item_roster(record['id']).items() if GROUPS[i['type']] == group}
        rows = record.get(group, [])
        ids = [row.get('id') for row in rows]
        duplicate = [key for key, n in collections.Counter(ids).items() if n > 1]
        missing, extra = sorted(set(roster) - set(ids)), sorted(set(ids) - set(roster))
        groups[group] = {'expected': len(roster), 'provided': len(rows), 'missing': missing, 'extra': extra, 'duplicates': duplicate}
        for n, row in enumerate(rows):
            path = group + '[' + str(n) + ']:' + str(row.get('id'))
            if row.get('id') in roster:
                item = roster[row['id']]
                expected_at = [item['first_line'], item['final_line']]
                at_valid = row.get('at') == expected_at if group == 'commands' else row.get('at') in (item['first_line'], expected_at)
                if not at_valid:
                    errors.append({'path': path, 'error': 'wrong original source event lines'})
                if not any(x in row.get('ev', []) for x in item['occurrence_lines']):
                    errors.append({'path': path, 'error': 'no evidence for this item'})
            try:
                source.evidence(record['id'], row)
            except (Invalid, KeyError, TypeError, ValueError) as error:
                errors.append({'path': path, 'error': str(error)})
    for n, link in enumerate(record.get('links', [])):
        try:
            source.evidence(record['id'], link, 'from', check_quote=False)
            source.evidence(record['id'], link, 'to', check_quote=False)
            source.evidence(record['id'], {'ev': sorted(set(link['from'] + link['to'])), **({'quote': link['quote']} if 'quote' in link else {})})
        except (Invalid, KeyError, TypeError, ValueError) as error:
            errors.append({'path': 'links[' + str(n) + ']', 'error': str(error)})
    for d in DIMS:
        for n, row in enumerate(record.get('portfolio', {}).get(d, [])):
            try:
                source.evidence(record['id'], row)
            except (Invalid, KeyError, TypeError, ValueError) as error:
                errors.append({'path': 'portfolio.' + d + '[' + str(n) + ']', 'error': str(error)})
    try:
        choice = record['choice']
        source.evidence(record['id'], choice, require_nonempty=any(choice.get(k) for k in ('stated', 'implemented', 'emphasized')))
    except (Invalid, KeyError, TypeError, ValueError) as error:
        errors.append({'path': 'choice', 'error': str(error)})
    complete_ids = all(not any(group[k] for k in ('missing', 'extra', 'duplicates')) for group in groups.values())
    return {'id': record['id'], 'all_items_covered': complete_ids, 'groups': groups, 'evidence_errors': errors,
            'source_sha256': p['eventstream_sha256'], 'source_events': len(lines), 'valid': complete_ids and not errors,
            'semantic_correctness_established': False}


def draft_coverage_check(record, source, allow_coverage_errors=False):
    """Bind source identity/hash; optional invalid-A coverage remains explicit."""
    p, (_, lines, _) = source.get(record['id'])
    require(record.get('hash') == p['eventstream_sha256'], 'draft source hash mismatch')
    counts, errors, expected_roster = {}, [], {}
    line_bindings = True
    for group in ('commands', 'messages', 'files'):
        roster = {key: i for key, i in source.item_roster(record['id']).items() if GROUPS[i['type']] == group}
        expected_roster[group] = list(roster)
        rows = record.get(group)
        require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), 'draft missing/malformed item group: ' + group)
        ids = [row.get('id') for row in rows]
        require(all(isinstance(key, str) for key in ids), 'draft item IDs must be strings')
        observed = collections.Counter(ids)
        defect = {'group': group, 'missing': sorted(set(roster) - set(ids)), 'extra': sorted(set(ids) - set(roster)),
                  'duplicates': {key: n for key, n in observed.items() if n > 1}}
        if defect['missing'] or defect['extra'] or defect['duplicates']:
            errors.append(defect)
        if not allow_coverage_errors:
            require(not errors, 'draft missing/duplicate/extra ' + group + ' IDs')
        for row in rows:
            if row['id'] not in roster:
                line_bindings = False; continue
            item = roster[row['id']]
            pair = [item['first_line'], item['final_line']]
            at_matches = row.get('at') == pair if group == 'commands' else row.get('at') in (item['first_line'], pair)
            line_bindings = line_bindings and at_matches
        counts[group] = len(rows)
    return {'source_hash_valid': True, 'all_item_ids_valid': not errors, 'source_line_bindings_valid': line_bindings,
            'counts': counts, 'source_events': len(lines), 'expected_source_roster': expected_roster,
            'coverage_errors': errors, 'coverage_errors_allowed_for_draft_only': allow_coverage_errors,
            'semantic_annotation_valid': False}


def canonicalize_derived(record, source, deduplicate_sets=False):
    """Reconcile derived metadata only; never assign semantic family/role labels."""
    draft_coverage_check(record, source)
    p, _ = source.get(record['id'])
    out = copy.deepcopy(record)
    roster = source.item_roster(record['id'])
    raw_derived = {'portfolio': copy.deepcopy(record.get('portfolio')),
                   'feedback': {r['id']: r.get('feedback') for r in record['commands']},
                   'positions': {r['id']: copy.deepcopy(r.get('at')) for g in ('commands', 'messages', 'files') for r in record[g]},
                   'link_orders': [link.get('order') for link in record.get('links', [])]}
    changes = []
    if deduplicate_sets:
        raw_derived['set_arrays'] = []
        for row in out['commands']:
            for field in ('families', 'dim', 'purpose'):
                values = row.get(field)
                require(isinstance(values, list) and all(isinstance(value, str) for value in values), 'set normalization requires explicit string arrays')
                raw_derived['set_arrays'].append({'item': row['id'], 'field': field, 'values': copy.deepcopy(values)})
                unique = list(dict.fromkeys(values))
                if unique != values:
                    changes.append({'item': row['id'], 'field': field, 'before': copy.deepcopy(values), 'after': unique,
                                    'normalization': 'lossless_set_deduplication', 'unique_labels_changed': False})
                row[field] = unique
    for group in ('commands', 'messages', 'files'):

        for row in out[group]:
            item = roster[row['id']]
            after = item['first_line'] if group == 'messages' else [item['first_line'], item['final_line']]
            if row.get('at') != after:
                changes.append({'item': row['id'], 'field': 'at', 'before': row.get('at'), 'after': after})
            row['at'] = after
    portfolio = {d: [] for d in DIMS}
    witnesses = {d: {} for d in DIMS}
    for row in out['commands']:
        if row.get('exec') not in ('evaluated', 'partial_results', 'mixed'):
            continue
        require(isinstance(row.get('families'), list), 'canonicalization needs explicit reviewed family labels')
        for token in row['families']:
            require(isinstance(token, str) and ':' in token, 'invalid reviewed family label')
            dim, family = token.split(':', 1)
            require(dim in DIMS, 'invalid reviewed family dimension')
            witnesses[dim].setdefault(family, []).append(row)
    for dim in DIMS:
        for family, rows in sorted(witnesses[dim].items()):
            first_row = min(rows, key=lambda r: r['at'][0])
            first = first_row['at'][1] if first_row['at'][1] is not None else first_row['at'][0]
            ev = sorted({line for row in rows for line in row['at'] if line is not None})
            portfolio[dim].append({'f': family, 'first': first, 'ev': ev, 'status': 'confirmed', 'reason': None})
        # Explicit unresolved identities are retained as uncertainty, never silently
        # promoted to confirmed families or discarded by a mechanical union.
        for entry in record.get('portfolio', {}).get(dim, []):
            if entry.get('status') == 'unresolved' and entry.get('f') not in witnesses[dim]:
                portfolio[dim].append(copy.deepcopy(entry))
        portfolio[dim].sort(key=lambda r: r['f'])
    if out.get('portfolio') != portfolio:
        changes.append({'field': 'portfolio', 'before': copy.deepcopy(out.get('portfolio')), 'after': portfolio})
    out['portfolio'] = portfolio
    feedback = derive_feedback(out)
    for row in out['commands']:
        if row.get('feedback') != feedback[row['id']]:
            changes.append({'item': row['id'], 'field': 'feedback', 'before': row.get('feedback'), 'after': feedback[row['id']]})
        row['feedback'] = feedback[row['id']]
    for n, link in enumerate(out.get('links', [])):
        frm, to = link.get('from'), link.get('to')
        require(isinstance(frm, list) and bool(frm) and all(integer(x) for x in frm), 'link.from needs explicit nonempty event lines')
        require(isinstance(to, list) and bool(to) and all(integer(x) for x in to), 'link.to needs explicit nonempty event lines')
        order = 'after' if max(frm) < min(to) else ('before' if max(to) < min(frm) else 'same_batch_unordered')
        if link.get('order') != order:
            changes.append({'field': 'links[' + str(n) + '].order', 'before': link.get('order'), 'after': order})
        link['order'] = order
    return out, {'schema': 'card-derived-metadata-reconciliation-v3', 'raw_model_derived': raw_derived,
                 'changes': changes, 'canonical_annotation_sha256': sha(canonical(out)),
                 'semantic_labels_changed': False, 'unique_label_sets_changed': False, 'set_deduplication_enabled': deduplicate_sets, 'source_positions_bound_from_item_ids': True,
                 'source_quote_length_establishes_semantic_accuracy': False}


def validation_snapshot(record, source):
    try:
        return validate(record, source)
    except (Invalid, KeyError, TypeError, ValueError) as error:
        return {'valid': False, 'error': str(error), 'semantic_correctness_established': False}


def semantic_changed_fields(before, after, group):
    fields = {'commands': ['purpose', 'main_purpose', 'dim', 'exec', 'families'],
              'messages': ['criterion', 'action', 'ref'], 'files': ['role', 'choice']}[group]
    unordered = {'purpose', 'dim', 'families', 'criterion', 'action', 'ref'}
    changed = []
    for field in fields:
        old, new = before.get(field), after.get(field)
        same = set(old) == set(new) if field in unordered and isinstance(old, list) and isinstance(new, list) else old == new
        if not same:
            changed.append(field)
    return changed


def derive_audit_verdict(before, after, group, supplied_decision):
    if before is None or isinstance(before, list):
        rows = [] if before is None else before
        derived = 'unresolved' if supplied_decision == 'unresolved' else 'correct'
        return {'supplied_decision': supplied_decision, 'derived_decision': derived, 'verdict_discrepancy': supplied_decision != derived,
                'prediction_row_state': 'missing' if not rows else 'ambiguous_duplicate', 'raw_prediction_rows': copy.deepcopy(rows),
                'semantic_fields_changed': None, 'evidence_fields_changed': None, 'semantic_correctness_established': False}
    changed = semantic_changed_fields(before, after, group)
    evidence_changed = [field for field in ('ev', 'quote') if before.get(field) != after.get(field)]
    derived = 'unresolved' if supplied_decision == 'unresolved' else ('correct' if changed or evidence_changed else 'confirm')
    return {'supplied_decision': supplied_decision, 'derived_decision': derived,
            'verdict_discrepancy': supplied_decision != derived,
            'semantic_fields_changed': changed, 'evidence_fields_changed': evidence_changed,
            'semantic_correctness_established': False}


def validate_audit_output(prediction, document, source, deduplicate_sets=False):
    prediction_coverage = draft_coverage_check(prediction, source, allow_coverage_errors=True)
    schema_validate(document, audit_output_schema(prediction, source))
    canonical_annotation, reconciliation = canonicalize_derived(document['annotation'], source, deduplicate_sets=deduplicate_sets)
    validate(canonical_annotation, source)
    unresolved_list(document['unresolved_adjudication'])
    pending = {x['id'] for x in document['unresolved_adjudication']}
    counts = {}
    discrepancies = []
    quote_matches = []
    for group in ('commands', 'messages'):
        for row in canonical_annotation[group]:
            quote_matches.append(dict(item_quote_check(canonical_annotation, row, source, group), path='annotation.' + group + '.' + row['id']))
    for n, link in enumerate(canonical_annotation.get('links', [])):
        quote_matches.append(dict(source.evidence(canonical_annotation['id'], {'ev': sorted(set(link['from'] + link['to'])), 'quote': link['quote']}), path='annotation.links[' + str(n) + ']'))
    semantic = {'commands': ['purpose', 'main_purpose', 'dim', 'exec', 'families'],
                'messages': ['criterion', 'action', 'ref'], 'files': ['role', 'choice']}
    for group in ('commands', 'messages', 'files'):
        before = {}
        for raw_row in prediction[group]: before.setdefault(raw_row['id'], []).append(raw_row)
        after = {row['id']: row for row in canonical_annotation[group]}
        rows = document['item_audit'][group]
        ids = [row['id'] for row in rows]
        require(len(ids) == len(set(ids)) and set(ids) == set(after), 'audit must explicitly cover every ' + group + ' ID exactly once')
        for row in rows:
            key = row['id']
            source.evidence(document['annotation']['id'], row, check_quote=False)
            own = source.item_roster(document['annotation']['id'])[key]['occurrence_lines']
            require(any(n in row['ev'] for n in own if n is not None), key + ': audit needs own item evidence')
            quote_matches.append(dict(item_quote_check(document['annotation'], row, source, group), path='item_audit.' + group + '.' + key))
            require(isinstance(row.get('reason'), str) and bool(row['reason'].strip()), key + ': audit reason must be nonempty and source-specific')
            raw_rows = before.get(key, [])
            prior = raw_rows[0] if len(raw_rows) == 1 else (raw_rows or None)
            verdict = derive_audit_verdict(prior, after[key], group, row['decision'])
            if row['decision'] == 'unresolved':
                require(key in pending, key + ': unresolved decision must remain in unresolved_adjudication')
            discrepancies.append({'id': key, 'group': group, 'decision': row['decision'], **verdict})
        counts[group] = len(rows)
    return {'valid': True, 'source_hash_valid': True, 'prediction_sha256': sha(canonical(prediction)),
            'all_item_audit_coverage': counts, 'discrepancies': discrepancies, 'prediction_coverage': prediction_coverage,
            'unresolved_count': len(pending), 'canonicalization': reconciliation, 'quote_matches': quote_matches, 'reason_semantic_quality_established': False,
            'semantic_correctness_established': False}


def whole_record_patch(before, after):
    patch = []
    for key in sorted(set(before) | set(after)):
        pointer = '/' + key.replace('~', '~0').replace('/', '~1')
        if key not in after:
            patch.append({'op': 'remove', 'path': pointer})
        elif key not in before:
            patch.append({'op': 'add', 'path': pointer, 'value': after[key]})
        elif before[key] != after[key]:
            patch.append({'op': 'replace', 'path': pointer, 'value': after[key]})
    return patch


def validate(record, source):
    if source.production:
        schema_validate(record, production_schema(source.vocab, source.extra_enums))
    require(isinstance(record, dict), 'annotation must be an object')
    for key in ('v', 'id', 'hash', 'commands', 'messages', 'files', 'portfolio', 'links', 'choice', 'coverage'):
        require(key in record, 'missing field: ' + key)
    require(record['v'] == 3, 'annotation version must be 3')
    p, (_, lines, _) = source.get(record['id'])
    require(record['hash'] == p['eventstream_sha256'], 'annotation/source hash mismatch')
    if source.production:
        require(record['id'] == (p.get('coding_id') or p['case_id']), 'production annotation must use neutral coding ID (original remains separate)')
    roster = source.item_roster(record['id'])
    seen = set()
    union = {d: set() for d in DIMS}
    for group in ('commands', 'messages', 'files'):
        rows = record[group]
        require(isinstance(rows, list), group + ': expected array')
        expected = {k for k, item in roster.items() if GROUPS[item['type']] == group}
        ids = [r.get('id') for r in rows]
        require(len(ids) == len(set(ids)), group + ': duplicate IDs')
        require(set(ids) == expected, group + ': missing/extra IDs ' + str(sorted(expected ^ set(ids))))
        for row in rows:
            key = row['id']
            require(key not in seen, 'item occurs in multiple groups')
            seen.add(key)
            item = roster[key]
            at = row.get('at')
            expected_at = [item['first_line'], item['final_line']]
            require(at == expected_at if group == 'commands' else at in (item['first_line'], expected_at), key + ': wrong original event lines')
            source.evidence(record['id'], row, check_quote=not (source.production and group in ('commands', 'messages')))
            if source.production and group in ('commands', 'messages'):
                item_quote_check(record, row, source, group)
            # Every row must cite one of its own events; other lines may support relations.
            require(any(n in row['ev'] for n in item['occurrence_lines']), key + ': no evidence for this item')
            if group == 'commands':
                enums(row.get('purpose'), PURPOSE, key + '.purpose')
                require(bool(row['purpose']), key + ': purpose required')
                if source.production or 'main_purpose' in row:
                    require(row.get('main_purpose') in PURPOSE + ['mixed'], key + ': main_purpose required/invalid')
                    require(row['main_purpose'] == 'mixed' or row['main_purpose'] in row['purpose'], key + ': main_purpose must occur in purpose labels')
                enums(row.get('dim'), DIMS, key + '.dim')
                require(row.get('exec') in EXEC, key + ': invalid exec enum')
                require(row.get('feedback') in FEEDBACK, key + ': invalid feedback enum')
                strings(row.get('families'), key + '.families')
                for token in row['families']:
                    require(':' in token, key + ': family must be dimension:family')
                    dim, family = token.split(':', 1)
                    require(family_allowed(dim, family, source), key + ': family not in vocabulary: ' + token)
                    require(dim in row['dim'], key + ': family dimension not recorded')
                    if row['exec'] in ('evaluated', 'partial_results', 'mixed'):
                        union[dim].add(family)
                if row['exec'] in ('none', 'definition_only', 'attempt_failed'):
                    require(not row['families'], key + ': unexecuted/failed command cannot contribute evaluated families')
            elif group == 'messages':
                enums(row.get('criterion'), CRITERIA, key + '.criterion')
                enums(row.get('action'), ACTIONS, key + '.action')
                strings(row.get('ref'), key + '.ref')
                require(set(row['ref']) <= set(roster), key + ': dangling item reference')
            else:
                require(isinstance(row.get('role'), str) and bool(row['role']), key + ': role required')
                require(isinstance(row.get('choice'), str), key + ': file choice must be a string (empty if no choice)')
                if 'file_role' in source.extra_enums:
                    require(row['role'] in source.extra_enums['file_role'], key + ': file role outside frozen enum')
            if 'states' in row:
                enums(row['states'], STATES, key + '.states')
                require(bool(row['states']), key + ': states cannot be empty')
                if len(row['states']) > 1:
                    # They are simultaneous unless explicit within-item ordering supplied.
                    require(row.get('state_order') in ('unordered', 'explicit'), key + ': state_order required')
                    if row['state_order'] == 'explicit':
                        require('order_evidence' in row, key + ': explicit order needs evidence')
                        source.evidence(record['id'], row['order_evidence'])
    require(isinstance(record['portfolio'], dict) and set(record['portfolio']) == set(DIMS), 'portfolio must contain exactly six dimensions')
    for dim in DIMS:
        rows = record['portfolio'][dim]
        require(isinstance(rows, list), 'portfolio.' + dim + ': array required')
        require(len({r.get('f') for r in rows}) == len(rows), 'duplicate portfolio family')
        confirmed = set()
        for row in rows:
            require(family_allowed(dim, row.get('f', ''), source), 'portfolio family outside vocabulary: ' + dim + ':' + str(row.get('f')))
            require(row.get('status') in ('confirmed', 'unresolved'), 'invalid portfolio status')
            require(integer(row.get('first')) and 1 <= row['first'] <= len(lines), 'invalid portfolio first event')
            source.evidence(record['id'], row)
            require(row['first'] in row['ev'], 'portfolio first must be evidence line')
            if row['status'] == 'confirmed':
                confirmed.add(row['f'])
                witnesses = [r for r in record['commands'] if r['exec'] in ('evaluated', 'partial_results', 'mixed') and dim + ':' + row['f'] in r['families']]
                require(bool(witnesses), 'confirmed family missing labeled evaluation')
                first = min(r['at'][0] for r in witnesses)
                require(first <= row['first'] <= min(r['at'][1] or r['at'][0] for r in witnesses if r['at'][0] == first), 'portfolio first outside first labeled evaluation')
            else:
                require(isinstance(row.get('reason'), str) and bool(row['reason']), 'unresolved family requires reason')
        require(confirmed == union[dim], 'portfolio differs from labeled evaluated union for ' + dim + ': ' + str(sorted(confirmed ^ union[dim])))
    require(isinstance(record['links'], list), 'links must be array')
    for link in record['links']:
        require(link.get('kind') in KINDS, 'invalid link kind')
        require(link.get('order') in ORDERS, 'invalid link order')
        require(link.get('status') in ('confirmed', 'unresolved'), 'invalid link status')
        enums(link.get('criterion'), CRITERIA, 'link.criterion')
        source.evidence(record['id'], link, 'from', check_quote=False)
        source.evidence(record['id'], link, 'to', check_quote=False)
        source.evidence(record['id'], {'ev': sorted(set(link['from'] + link['to'])), **({'quote': link['quote']} if 'quote' in link else {})})
        require(isinstance(link.get('before'), str) and isinstance(link.get('after'), str), 'link before/after strings required')
        if link['order'] == 'after':
            require(max(link['from']) < min(link['to']), 'after link contradicts source line order')
        if link['order'] == 'before':
            require(max(link['to']) < min(link['from']), 'before link contradicts source line order')
    choice = record['choice']
    for field in ('stated', 'implemented', 'emphasized'):
        strings(choice.get(field), 'choice.' + field)
    require(isinstance(choice.get('relation'), str) and bool(choice['relation']), 'choice relation required')
    if 'choice_relation' in source.extra_enums:
        require(choice['relation'] in source.extra_enums['choice_relation'], 'choice relation outside frozen enum')
    source.evidence(record['id'], choice, require_nonempty=any(choice[k] for k in ('stated', 'implemented', 'emphasized')))
    coverage = record['coverage']
    require(type(coverage.get('complete')) is bool, 'coverage.complete must be boolean')
    strings(coverage.get('unknown_items'), 'coverage.unknown_items')
    require(set(coverage['unknown_items']) <= set(roster), 'unknown_items outside roster')
    require(isinstance(coverage.get('unavailable'), list), 'coverage.unavailable must be array')
    return {'valid': True, 'id': record['id'], 'source_hash': record['hash'], 'items': len(roster),
            'item_counts': {g: len(record[g]) for g in ('commands', 'messages', 'files')},
            'source_events': len(lines), 'annotation_coverage_complete': coverage['complete'],
            'semantic_correctness_established': False}


def template(identifier, source):
    p, _ = source.get(identifier)
    x = {'v': 3, 'id': p.get('coding_id') or p['case_id'], 'hash': p['eventstream_sha256'],
         'commands': [], 'messages': [], 'files': [], 'portfolio': {d: [] for d in DIMS}, 'links': [],
         'choice': {'stated': [], 'implemented': [], 'emphasized': [], 'relation': None, 'ev': []},
         'coverage': {'complete': False, 'unknown_items': [], 'unavailable': []}}
    for i in source.item_roster(identifier).values():
        g = GROUPS[i['type']]
        row = {'id': i['item_id'], 'at': [i['first_line'], i['final_line']] if g == 'commands' else i['first_line'], 'ev': []}
        if g == 'commands':
            row.update(purpose=None, main_purpose=None, dim=None, exec=None, families=None, feedback=None, quote=None)
        elif g == 'messages':
            row.update(criterion=None, action=None, ref=None, quote=None)
        else:
            row.update(role=None, choice=None)
        x[g].append(row)
    return x


def derive_feedback(record):
    """Derive recurrence solely from semantic family/execution labels and event order.

    Earlier numerical availability requires an earlier completed evaluation label.
    Inference-only additions never create substantive expansion. Same-batch
    exploration needs two different families in at least one substantive dimension.
    No inspected/causal response is inferred from chronology.
    """
    out = {}
    rows = sorted(record['commands'], key=lambda x: x['at'][0])
    for row in rows:
        if row['exec'] == 'unknown':
            out[row['id']] = 'unknown'
            continue
        if row['exec'] not in ('evaluated', 'partial_results', 'mixed'):
            out[row['id']] = 'no'
            continue
        earlier = [x for x in rows if x['exec'] in ('evaluated', 'partial_results', 'mixed') and x['at'][1] is not None and x['at'][1] < row['at'][0]]
        prior = {f for x in earlier for f in x['families'] if f.split(':', 1)[0] != 'I'}
        current = {f for f in row['families'] if f.split(':', 1)[0] != 'I'}
        new = current - prior
        multiple = any(sum(f.startswith(d + ':') for f in current) >= 2 for d in DIMS[:-1])
        if earlier and new:
            out[row['id']] = 'new_family_after_result'
        elif new and multiple:
            out[row['id']] = 'new_family_same_batch'
        elif earlier and current & prior:
            out[row['id']] = 'revisit_after_result'
        else:
            out[row['id']] = 'no'
    return out


def review_flags(record, source):
    _, (_, _, events) = source.get(record['id'])
    flags = []
    for row in record['commands']:
        if row['exec'] == 'evaluated' and row['at'][1] is not None:
            item = events[row['at'][1] - 1].get('item', {})
            code = item.get('exit_code')
            if integer(code) and code != 0:
                flags.append({'id': row['id'], 'flag': 'nonzero_exit_with_evaluated_label', 'exit_code': code,
                              'action': 'source review required; label preserved because partial fit may precede unrelated failure'})
    for row in record['commands']:
        for token in row['families']:
            if token.startswith('M:') and token[2:] not in PRODUCTION_FAMILIES['M']:
                flags.append({'id': row['id'], 'flag': 'legacy_or_noncanonical_mean_family_label', 'provided': token,
                              'action': 'review against canonical production grammar; no automatic recoding'})
    return flags


def schema_validate(value, schema, path='$'):
    """Validate the small JSON-Schema subset emitted here, without dependencies."""
    checks = {'object': lambda x: isinstance(x, dict), 'array': lambda x: isinstance(x, list),
              'string': lambda x: isinstance(x, str), 'integer': integer,
              'boolean': lambda x: type(x) is bool, 'null': lambda x: x is None}
    types = schema.get('type')
    types = [types] if isinstance(types, str) else types
    if types:
        require(any(checks[t](value) for t in types), path + ': wrong JSON type')
    if 'enum' in schema:
        require(value in schema['enum'], path + ': outside allowed enum')
    if isinstance(value, dict):
        props = schema.get('properties', {})
        require(set(schema.get('required', [])) <= set(value), path + ': missing required fields ' + str(sorted(set(schema.get('required', [])) - set(value))))
        if schema.get('additionalProperties') is False:
            require(set(value) <= set(props), path + ': unexpected properties ' + str(sorted(set(value) - set(props))))
        for k in props:
            if k in value:
                schema_validate(value[k], props[k], path + '.' + k)
    elif isinstance(value, list):
        require(len(value) >= schema.get('minItems', 0), path + ': too few array values')
        require(len(value) <= schema.get('maxItems', len(value)), path + ': too many array values')
        for n, item in enumerate(value):
            schema_validate(item, schema.get('items', {}), path + '[' + str(n) + ']')
    elif isinstance(value, str) and 'minLength' in schema:
        require(len(value) >= schema['minLength'], path + ': empty string')
    elif integer(value) and 'minimum' in schema:
        require(value >= schema['minimum'], path + ': below minimum')


def production_schema(vocabulary=None, extra_enums=None):
    """Strict JSON Schema suitable for constrained native CLI output."""
    vocabulary = vocabulary or PRODUCTION_FAMILIES
    extra_enums = extra_enums or {}
    def obj(props):
        return {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}
    def arr(item):
        return {'type': 'array', 'items': item}
    def enum(values):
        return {'type': 'string', 'enum': list(values)}
    string = {'type': 'string'}
    line = {'type': 'integer', 'minimum': 1}
    evidence = arr(line)
    # Tuple schemas are avoided for compatibility with constrained-output engines;
    # validate() enforces exactly [source_start, source_completion_or_null].
    at = {'type': 'array', 'items': {'type': ['integer', 'null']}, 'minItems': 2, 'maxItems': 2}
    command = obj({'id': string, 'at': at, 'purpose': arr(enum(PURPOSE)), 'main_purpose': enum(PURPOSE + ['mixed']),
                   'dim': arr(enum(DIMS)), 'exec': enum(EXEC), 'families': arr(enum([d + ':' + f for d in DIMS for f in vocabulary[d]])),
                   'feedback': enum(FEEDBACK), 'ev': evidence, 'quote': {'type': 'string', 'minLength': 1}})
    message = obj({'id': string, 'at': line, 'criterion': arr(enum(CRITERIA)), 'action': arr(enum(ACTIONS)), 'ref': arr(string), 'ev': evidence, 'quote': {'type': 'string', 'minLength': 1}})
    file = obj({'id': string, 'at': at, 'role': enum(extra_enums['file_role']) if 'file_role' in extra_enums else string, 'choice': string, 'ev': evidence})
    # Nullable reason/quote fields keep strict schemas complete without inventing text.
    portfolio = obj({d: arr(obj({'f': enum(vocabulary[d]), 'first': line, 'ev': evidence, 'status': enum(['confirmed', 'unresolved']),
                                'reason': {'type': ['string', 'null']}})) for d in DIMS})
    link = obj({'kind': enum(KINDS), 'from': evidence, 'to': evidence, 'criterion': arr(enum(CRITERIA)), 'before': string,
                'after': string, 'order': enum(ORDERS), 'status': enum(['confirmed', 'unresolved']), 'quote': {'type': 'string', 'minLength': 1}})
    return obj({'v': {'type': 'integer', 'enum': [3]}, 'id': string, 'hash': string,
                'commands': arr(command), 'messages': arr(message), 'files': arr(file), 'portfolio': portfolio, 'links': arr(link),
                'choice': obj({'stated': arr(string), 'implemented': arr(string), 'emphasized': arr(string), 'relation': enum(extra_enums['choice_relation']) if 'choice_relation' in extra_enums else string, 'ev': evidence}),
                'coverage': obj({'complete': {'type': 'boolean'}, 'unknown_items': arr(string), 'unavailable': arr(string)})})


def bound_annotation_schema(identifier, source):
    packet, _ = source.get(identifier)
    schema = production_schema(source.vocab, source.extra_enums)
    schema['properties']['id'] = {'type': 'string', 'enum': [packet.get('coding_id') or packet['case_id']]}
    schema['properties']['hash'] = {'type': 'string', 'enum': [packet['eventstream_sha256']]}
    for group in ('commands', 'messages', 'files'):
        ids = [i['item_id'] for i in packet['items'] if GROUPS[i['type']] == group]
        array = schema['properties'][group]
        array.update(minItems=len(ids), maxItems=len(ids))
        array['items']['properties']['id'] = {'type': 'string', 'enum': ids} if ids else {'type': 'string'}
    return schema


def audit_output_schema(prediction=None, source=None):
    """Strict full-record correction plus source-backed item decisions.

    A concrete prediction locks identity, source and prediction hash from the host.
    No confirmation or quote is generated by this schema.
    """
    def obj(props):
        return {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}
    def arr(item):
        return {'type': 'array', 'items': item}
    string = {'type': 'string'}
    nonempty = {'type': 'string', 'minLength': 1}
    line = {'type': 'integer', 'minimum': 1}
    annotation = production_schema(source.vocab if source else None, source.extra_enums if source else None)
    source_hash = string
    prediction_hash = string
    identity = string
    packet = None
    if prediction is not None:
        prediction_hash = {'type': 'string', 'enum': [sha(canonical(prediction))]}
        require(source is not None, 'bound audit schema requires source')
        packet, _ = source.get(prediction['id'])
        canonical_id = packet.get('coding_id') or packet['case_id']
        identity = {'type': 'string', 'enum': [canonical_id]}
        source_hash = {'type': 'string', 'enum': [packet['eventstream_sha256']]}
        annotation = bound_annotation_schema(prediction['id'], source)
    groups = {}
    for group in ('commands', 'messages', 'files'):
        ids = [i['item_id'] for i in packet['items'] if GROUPS[i['type']] == group] if packet else None
        row = obj({'id': {'type': 'string', 'enum': ids} if ids else string,
                   'decision': {'type': 'string', 'enum': ['confirm', 'correct', 'unresolved']},
                   'quote': nonempty, 'reason': nonempty, 'ev': arr(line)})
        groups[group] = arr(row)
        if ids is not None:
            groups[group].update(minItems=len(ids), maxItems=len(ids))
    unresolved = obj({'id': nonempty, 'reason': nonempty, 'status': {'type': 'string', 'enum': ['unresolved']}})
    return obj({'id': identity, 'source_sha256': source_hash, 'prediction_sha256': prediction_hash,
                'annotation': annotation, 'item_audit': obj(groups), 'notes': arr(nonempty),
                'unresolved_adjudication': arr(unresolved)})


def features(r, source):
    validate(r, source)
    confirmed = {d: sorted(x['f'] for x in r['portfolio'][d] if x['status'] == 'confirmed') for d in DIMS}
    unresolved = {d: sorted(x['f'] for x in r['portfolio'][d] if x['status'] == 'unresolved') for d in DIMS}
    minimum = sum(len(confirmed[d]) >= 2 for d in DIMS[:-1])
    maximum = sum(len(set(confirmed[d] + unresolved[d])) >= 2 for d in DIMS[:-1])
    if not r['coverage']['complete'] or r['coverage']['unknown_items'] or r['coverage']['unavailable']:
        maximum = 5
    pm = {'setup': 'prepare/inspect', 'inspect': 'prepare/inspect', 'construct': 'prepare/inspect',
          'evaluate': 'evaluate/surface', 'interpret': 'interpret', 'choose': 'choose/revise', 'revise': 'choose/revise',
          'verify': 'verify', 'report': 'report', 'debug': 'mixed/unresolved', 'other': 'mixed/unresolved', 'unknown': 'mixed/unresolved'}
    am = {'primary_adopt': 'choose/revise', 'primary_reject': 'choose/revise', 'primary_revise': 'choose/revise',
          'report_emphasis': 'report', 'next_analysis': 'mixed/unresolved', 'none': 'mixed/unresolved'}
    feedback_derived = derive_feedback(r)
    timeline = []
    for group in ('commands', 'messages', 'files'):
        for row in r[group]:
            at = row['at'][0] if isinstance(row['at'], list) else row['at']
            states = row.get('states')
            if not states:
                if group == 'commands':
                    states = [pm[row['main_purpose']]] if row.get('main_purpose') in pm else (['mixed/unresolved'] if row.get('main_purpose') == 'mixed' else sorted({pm[p] for p in row['purpose']}))
                elif group == 'messages':
                    states = sorted({am[p] for p in row['action']}) or ['mixed/unresolved']
                else:
                    states = ['mixed/unresolved']
            if len(states) > 1 and row.get('state_order') != 'explicit':
                states = ['mixed/unresolved']
            for order, state in enumerate(states):
                timeline.append({'at': at, 'within_item': order, 'id': row['id'], 'state': state,
                                 'feedback': feedback_derived.get(row['id'], 'no')})
    timeline.sort(key=lambda x: (x['at'], x['within_item']))
    collapsed = []
    for x in timeline:
        if not collapsed or collapsed[-1]['state'] != x['state']:
            collapsed.append(copy.deepcopy(x))
        else:
            collapsed[-1].setdefault('collapsed_ids', []).append(x['id'])
    transitions = sorted({(a['state'], b['state']) for a, b in zip(collapsed, collapsed[1:])})
    family_occurrences = collections.Counter(f for c in r['commands'] for f in c['families'])
    # No source text is inspected for these features; all semantics are labels.
    choices = [x['at'][0] for x in r['commands'] if set(x['purpose']) & {'choose', 'revise'}]
    choices += [x['at'] if isinstance(x['at'], int) else x['at'][0] for x in r['messages'] if set(x['action']) & {'primary_adopt', 'primary_revise'}]
    last_choice = max(choices, default=None)
    return {'id': r['id'], 'B_min': minimum, 'B_max': maximum, 'families': confirmed,
            'unresolved_families': unresolved, 'family_counts': {d: len(confirmed[d]) for d in DIMS},
            'repeated_family_labels': {f: n for f, n in sorted(family_occurrences.items()) if n > 1},
            'feedback_derived': feedback_derived,
            'feedback_counts': dict(collections.Counter(feedback_derived.values())),
            'feedback_discrepancies': [{'id': row['id'], 'provided': row['feedback'], 'derived': feedback_derived[row['id']]} for row in r['commands'] if row['feedback'] != feedback_derived[row['id']]],
            'review_flags': review_flags(r, source),
            'timeline': timeline, 'collapsed_timeline': collapsed, 'case_nodes': sorted({x['state'] for x in timeline}),
            'case_transitions': [list(t) for t in transitions], 'first_explicit_choice': min(choices, default=None),
            'last_explicit_choice_or_revision': last_choice,
            'verification_after_last_choice': None if last_choice is None else any('verify' in c['purpose'] and c['at'][0] > last_choice for c in r['commands']),
            'semantic_correctness_established': False}


def receipt_check(receipt, record, source, audit=False, draft=False):
    require(isinstance(receipt, dict), 'receipt object required')
    p, (_, lines, _) = source.get(record['id'])
    require(receipt.get('source_sha256') == record['hash'], 'receipt source hash mismatch')
    require(receipt.get('full_source_read') is True, 'explicit complete source-read attestation required')
    if source.production:
        require(receipt.get('context_case_count') == 1, 'production requires exactly one case per model context')
    require(receipt.get('predictions_visible') is audit, 'prediction visibility mismatch')
    covered = receipt.get('covered_items', {})
    for g in ('commands', 'messages', 'files'):
        strings(covered.get(g), 'receipt.covered_items.' + g)
        require(set(covered[g]) == {key for key, item in source.item_roster(record['id']).items() if GROUPS[item['type']] == g}, 'receipt does not cover every source ' + g + ' item')
    ranges = receipt.get('event_line_ranges')
    require(isinstance(ranges, list) and bool(ranges), 'receipt.event_line_ranges required')
    seen = set()
    for pair in ranges:
        require(isinstance(pair, list) and len(pair) == 2 and all(integer(v) for v in pair), 'event ranges must be [first,last]')
        lo, hi = pair
        require(1 <= lo <= hi <= len(lines), 'event range out of source bounds')
        new = set(range(lo, hi + 1))
        require(not seen & new, 'overlapping event ranges')
        seen.update(new)
    require(seen == set(range(1, len(lines) + 1)), 'receipt omits source event lines')
    for k in ('reviewer', 'model', 'effort', 'tier', 'prompt_sha256'):
        require(isinstance(receipt.get(k), str) and bool(receipt[k]), 'receipt.' + k + ' required')
    require(receipt['tier'] in ('default', 'standard'), 'standard/default tier required')
    require(len(receipt['prompt_sha256']) == 64 and all(c in '0123456789abcdef' for c in receipt['prompt_sha256']), 'prompt_sha256 must be a SHA-256 digest')
    if audit:
        require(receipt.get('negative_items_checked') is True, 'audit must explicitly check absent/negative labels')
    require(isinstance(receipt.get('token_usage'), dict) and bool(receipt['token_usage']), 'actual token_usage object required')
    require(isinstance(receipt.get('cost'), dict) and bool(receipt['cost']), 'cost receipt object required (unavailable explicitly allowed)')
    for accounting in ('token_usage', 'cost'):
        values = receipt[accounting]
        if 'unavailable' in values:
            require(isinstance(values['unavailable'], str) and bool(values['unavailable']), 'receipt.' + accounting + ': unavailable needs an explicit reason')
        else:
            require(all(value is not None for value in values.values()), 'receipt.' + accounting + ': null accounting requires an unavailable reason')
    if not draft:
        require(record['coverage']['complete'], 'full-pass record must attest complete annotation coverage')


def patch_apply(document, patch, source=None):
    """Small strict RFC6902 implementation; never evaluates a patch value."""
    require(isinstance(patch, list), 'patch must be array')
    out = copy.deepcopy(document)
    for op in patch:
        require(op.get('op') in ('add', 'replace', 'remove', 'test'), 'supported patch operations: add replace remove test')
        path = op.get('path')
        require(isinstance(path, str) and path.startswith('/'), 'JSON pointer required; root replacement disallowed')
        parts = [s.replace('~1', '/').replace('~0', '~') for s in path[1:].split('/')]
        node = out
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node[part]
        key = parts[-1]
        if isinstance(node, list):
            idx = len(node) if key == '-' else int(key)
            require(0 <= idx <= len(node), 'patch array index out of range')
            if op['op'] == 'add':
                node.insert(idx, copy.deepcopy(op['value']))
            else:
                require(idx < len(node), 'patch array element absent')
                if op['op'] == 'test':
                    require(node[idx] == op['value'], 'patch test failed')
                elif op['op'] == 'remove':
                    del node[idx]
                else:
                    node[idx] = copy.deepcopy(op['value'])
        else:
            require(isinstance(node, dict), 'patch target must be object or array')
            if op['op'] != 'add':
                require(key in node, 'patch object member absent')
            if op['op'] == 'test':
                require(node[key] == op['value'], 'patch test failed')
            elif op['op'] == 'remove':
                del node[key]
            else:
                node[key] = copy.deepcopy(op['value'])
    require(out.get('hash') == document.get('hash'), 'patch cannot change identity/source')
    if out.get('id') != document.get('id'):
        require(source is not None and out.get('id') in source.by_id and document.get('id') in source.by_id and source.by_id[out['id']]['case_id'] == source.by_id[document['id']]['case_id'], 'patch cannot change identity/source')
    return out


def differences(a, b):
    diff = {}
    for g in ('commands', 'messages', 'files'):
        aa, bb = ({r['id']: r for r in x[g]} for x in (a, b))
        rows = []
        for key in sorted(set(aa) | set(bb)):
            fields = sorted(k for k in set(aa.get(key, {})) | set(bb.get(key, {})) if aa.get(key, {}).get(k) != bb.get(key, {}).get(k))
            if fields:
                rows.append({'id': key, 'changed_fields': fields, 'A': aa.get(key), 'B': bb.get(key)})
        diff[g] = rows
    for field in ('portfolio', 'links', 'choice', 'coverage'):
        if a[field] != b[field]:
            diff[field] = {'A': a[field], 'B': b[field]}
    return {'id': a['id'], 'discrepancies': diff, 'interpretation': 'Structural discrepancies only; neither annotation is declared ground truth.'}


def manifest(source, batch_size):
    require(batch_size > 0, 'positive batch size required')
    replays = sorted((r for r in source.index if not r['is_original']), key=lambda r: r['case_id'])
    original = sorted((r for r in source.index if r['is_original']), key=lambda r: r['case_id'])
    require(len(replays) == 3000 and len(original) == 1, 'expected 3000 replays plus one separate original')
    # Assignment hash excludes labels and mutable review state.
    cohorts = []
    for n, r in enumerate(replays):
        cohorts.append({'id': r.get('coding_id') or r['case_id'], 'source_case_id': r['case_id'],
                        'source_sha256': r['eventstream_sha256'], 'packet_sha256': r['packet_sha256'],
                        'batch': f'replay-{n // batch_size + 1:03d}', 'position': n % batch_size + 1})
    return {'schema': 'card-full-cohort-review-manifest-v3', 'replay_count': 3000, 'original_count': 1,
            'batch_size': batch_size, 'ordering': 'lexicographic source case id; no strata or external labels',
            'assignment_sha256': sha(canonical(cohorts)), 'replays': cohorts,
            'original_separate': [{'id': r.get('coding_id') or r['case_id'], 'source_case_id': r['case_id'],
                                   'source_sha256': r['eventstream_sha256'], 'packet_sha256': r['packet_sha256'], 'batch': 'original-separate'} for r in original]}


def safe_dir(root, record, source):
    p, _ = source.get(record['id'])
    key = p.get('coding_id') or p['case_id']
    require('/' not in key and key not in ('.', '..'), 'invalid storage ID')
    return pathlib.Path(root) / key


def unresolved_list(value):
    require(isinstance(value, list), 'unresolved adjudication must be an explicit list')
    for row in value:
        require(isinstance(row, dict) and isinstance(row.get('id'), str) and isinstance(row.get('reason'), str) and bool(row['reason']), 'dispute id/reason required')
        require(row.get('status') == 'unresolved', 'pending dispute status must be unresolved')
    require(len({x['id'] for x in value}) == len(value), 'duplicate dispute IDs')


def status(source, store):
    expected = manifest(source, 25)
    result = {'expected_replays': 3000, 'expected_original_separate': 1, 'replays': {}, 'original_separate': {}, 'errors': []}
    for cohort, field in ((expected['replays'], 'replays'), (expected['original_separate'], 'original_separate')):
        counts = collections.Counter()
        for row in cohort:
            directory = pathlib.Path(store) / row['id']
            paths = {k: directory / f'{v}.json' for k, v in [('A', 'pass-A'), ('B', 'source-audit'), ('final', 'final')]}
            for k, p in paths.items():
                if not p.exists():
                    counts[k + '_missing'] += 1
                    continue
                try:
                    artifact = load(p)
                    annotation = artifact['annotation'] if k in ('A', 'final') else artifact['after']
                    require(source.by_id[annotation['id']]['case_id'] == row['source_case_id'], 'stored identity mismatch')
                    is_draft = k == 'A' and artifact.get('state') == 'draft_invalid'
                    if is_draft:
                        draft_coverage_check(annotation, source, allow_coverage_errors=artifact.get('draft_coverage_errors_allowed', False))
                        require(artifact.get('validation', {}).get('valid') is False, 'draft must preserve explicit invalid validation state')
                        require(json.loads(artifact['raw_annotation_text']) == annotation, 'draft raw annotation differs from saved parsed record')
                        require(sha(artifact['raw_annotation_text'].encode()) == artifact['raw_annotation_sha256'], 'draft raw byte hash mismatch')
                    else:
                        validate(annotation, source)
                    require(sha(canonical(annotation)) == artifact['annotation_sha256'], 'artifact annotation hash mismatch')
                    if k == 'A':
                        receipt_check(artifact['receipt'], annotation, source, draft=is_draft)
                        counts['A_source_complete'] += 1
                        if is_draft:
                            counts['A_draft'] += 1
                            continue
                    elif k == 'B':
                        receipt_check(artifact['receipt'], annotation, source, audit=True)
                        require(paths['A'].exists(), 'audit without pass A')
                        original = load(paths['A'])
                        require(artifact['before'] == original['annotation'], 'audit before differs from immutable A')
                        require(artifact['prediction_sha256'] == original['annotation_sha256'], 'audit prediction hash mismatch')
                        require(patch_apply(artifact['before'], artifact['patch'], source=source) == annotation, 'audit patch/after mismatch')
                        if source.production:
                            require('audit_output' in artifact, 'production audit lacks item-by-item decisions')
                            validate_audit_output(artifact['before'], artifact['audit_output'], source, deduplicate_sets=artifact.get('canonicalization', {}).get('set_deduplication_enabled', False))
                            corrected, _ = canonicalize_derived(artifact['audit_output']['annotation'], source, deduplicate_sets=artifact.get('canonicalization', {}).get('set_deduplication_enabled', False))
                            require(corrected == annotation, 'stored audit differs from canonical full audit output')
                    else:
                        require(paths['A'].exists() and paths['B'].exists(), 'final missing pass A or source audit')
                        aa, bb = load(paths['A']), load(paths['B'])
                        require(artifact['artifacts']['pass_A_sha256'] == sha(canonical(aa)), 'final pass A artifact hash mismatch')
                        require(artifact['artifacts']['source_audit_sha256'] == sha(canonical(bb)), 'final audit artifact hash mismatch')
                        expected_annotation = bb['after']
                        expected_pending = bb['unresolved_adjudication']
                        adj_path = directory / 'adjudication.json'
                        if adj_path.exists():
                            adj = load(adj_path)
                            require(artifact['artifacts'].get('adjudication_sha256') == sha(canonical(adj)), 'final adjudication hash mismatch')
                            require(adj['decision']['before_sha256'] == bb['annotation_sha256'], 'adjudication source mismatch')
                            require(patch_apply(bb['after'], adj['decision']['patch']) == adj['after'], 'adjudication patch mismatch')
                            expected_annotation, expected_pending = adj['after'], adj['decision']['unresolved_adjudication']
                        require(annotation == expected_annotation, 'final differs from audited/adjudicated annotation')
                        require(artifact['unresolved_adjudication'] == expected_pending, 'final drops unresolved disputes')
                        unresolved_list(artifact['unresolved_adjudication'])
                        require(artifact['status'] == ('unresolved' if expected_pending else 'audited'), 'final status contradicts adjudication')
                        counts['unresolved_final'] += bool(artifact['unresolved_adjudication'])
                    counts[k + '_valid'] += 1
                except (Invalid, KeyError, TypeError, ValueError) as e:
                    counts[k + '_invalid'] += 1
                    result['errors'].append({'id': row['id'], 'stage': k, 'error': str(e)})
        result[field] = dict(counts)
    result['full_cohort_complete'] = all(result[f].get('A_source_complete', 0) == n and result[f].get('B_valid', 0) == n and result[f].get('final_valid', 0) == n and not result[f].get('unresolved_final', 0) for f, n in [('replays', 3000), ('original_separate', 1)])
    result['semantic_correctness_established'] = False
    return result


def records(path):
    text = pathlib.Path(path).read_text()
    try:
        x = json.loads(text)
        return x if isinstance(x, list) else [x]
    except json.JSONDecodeError:
        return [json.loads(l) for l in text.splitlines() if l.strip()]


class PacketStore(Sources):
    """Preload once, then perform concurrent read-only source-bound operations.

    preload_manifest binds the exact complete frozen 3,000+original manifest.
    preload_ids is for bounded calibration/tests. Archive decompression happens
    once per preload, never once per case after the store is sealed. Returned
    source objects are shared read-only references; callers must not mutate them.
    """
    def __init__(self, packet_directory=None, raw_archive=None, vocabulary=None, production=True):
        super().__init__(packet_directory, raw_archive, vocabulary)
        self.production = production
        if production and not vocabulary:
            self.vocab = copy.deepcopy(PRODUCTION_FAMILIES)
        self._sealed = False
        self.epoch = None

    def collect(self, identifiers):
        if self._sealed:
            for identifier in identifiers:
                require(identifier in self.by_id, 'unknown case in frozen store: ' + str(identifier))
                require(self.by_id[identifier]['case_id'] in self._packets, 'case outside preloaded epoch: ' + str(identifier))
            return
        super().collect(identifiers)

    def preload_ids(self, identifiers):
        require(not self._sealed, 'PacketStore already sealed')
        ids = list(identifiers)
        super().collect(ids)
        self._common = load(self.directory / 'COMMON_OUTPUTS.json')
        for key, value in self._common.items():
            require(sha(value['text'].encode()) == key, 'common-output epoch hash mismatch')
        bindings = sorted((self.by_id[i]['case_id'], self.by_id[i]['eventstream_sha256'], self.by_id[i]['packet_sha256']) for i in ids)
        require(len({row[0] for row in bindings}) == len(bindings), 'duplicate case in PacketStore preload')
        self.epoch = {'schema': 'card-source-binding-epoch-v3', 'cases': len(bindings),
                      'bindings_sha256': sha(canonical(bindings)),
                      'source_index_sha256': sha((self.directory / 'INDEX.jsonl').read_bytes()),
                      'common_outputs_sha256': sha((self.directory / 'COMMON_OUTPUTS.json').read_bytes())}
        self._sealed = True
        return self

    def preload_manifest(self, manifest_path=None):
        path = pathlib.Path(manifest_path or HERE / 'FULL_COHORT_MANIFEST.json')
        frozen = load(path)
        expected = manifest(self, frozen['batch_size'])
        require(frozen == expected, 'frozen cohort manifest does not match current exact source index')
        rows = frozen['replays'] + frozen['original_separate']
        self.preload_ids([row['id'] for row in rows])
        self.epoch.update(manifest_sha256=sha(path.read_bytes()), full_cohort=True)
        return self


# Stable public API. These operations never mutate caller records or the store.
def validate_record(record, store):
    return validate(record, store)


def canonicalize_record(record, store, deduplicate_sets=False):
    return canonicalize_derived(record, store, deduplicate_sets=deduplicate_sets)


def validate_audit(prediction, audit_output, store, deduplicate_sets=False):
    return validate_audit_output(prediction, audit_output, store, deduplicate_sets=deduplicate_sets)


def draft_admissibility(record, store, allow_coverage_errors=False):
    return draft_coverage_check(record, store, allow_coverage_errors=allow_coverage_errors)


class _ModuleAPI:
    def __getattr__(self, name):
        return globals()[name]


_ERROR_API = _ModuleAPI()
_ERROR_SPEC = importlib.util.spec_from_file_location('card_citation_errors_v3', HERE / 'citation_errors.py')
_ERROR_HELPERS = importlib.util.module_from_spec(_ERROR_SPEC)
_ERROR_SPEC.loader.exec_module(_ERROR_HELPERS)


def collect_record_errors(record, store, canonicalize=True, deduplicate_sets=False):
    return _ERROR_HELPERS.record_errors(_ERROR_API, record, store, canonicalize=canonicalize, deduplicate_sets=deduplicate_sets)


def collect_audit_errors(prediction, audit_output, store, deduplicate_sets=False):
    return _ERROR_HELPERS.audit_errors(_ERROR_API, prediction, audit_output, store, deduplicate_sets=deduplicate_sets)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production', action='store_true', help='Require neutral IDs and dominant main_purpose')
    p.add_argument('--packets', type=pathlib.Path)
    p.add_argument('--raw', type=pathlib.Path)
    p.add_argument('--vocabulary', type=pathlib.Path, help='Explicit calibrated/frozen family vocabulary and optional enums')
    sub = p.add_subparsers(dest='command', required=True)
    x = sub.add_parser('read'); x.add_argument('ids', nargs='+'); x.add_argument('--output', type=pathlib.Path, required=True); x.add_argument('--raw-events', action='store_true')
    x = sub.add_parser('template'); x.add_argument('id'); x.add_argument('--output', type=pathlib.Path, required=True)
    x = sub.add_parser('manifest'); x.add_argument('--batch-size', type=int, default=25); x.add_argument('--output', type=pathlib.Path, required=True)
    for name in ('validate', 'features', 'coverage', 'canonicalize', 'errors'):
        x = sub.add_parser(name); x.add_argument('annotations', type=pathlib.Path)
    x = sub.add_parser('diff'); x.add_argument('A', type=pathlib.Path); x.add_argument('B', type=pathlib.Path); x.add_argument('--exposure-note')
    x = sub.add_parser('receipt-template'); x.add_argument('id'); x.add_argument('--output', type=pathlib.Path, required=True)
    x = sub.add_parser('record-a'); x.add_argument('annotation', type=pathlib.Path); x.add_argument('receipt', type=pathlib.Path); x.add_argument('--allow-invalid-draft', action='store_true'); x.add_argument('--allow-coverage-errors', action='store_true'); x.add_argument('--store', type=pathlib.Path, required=True)
    x = sub.add_parser('audit-errors'); x.add_argument('prediction', type=pathlib.Path); x.add_argument('audit_output', type=pathlib.Path)
    x = sub.add_parser('validate-audit'); x.add_argument('prediction', type=pathlib.Path); x.add_argument('audit_output', type=pathlib.Path)
    x = sub.add_parser('record-audit-output'); x.add_argument('id'); x.add_argument('audit_output', type=pathlib.Path); x.add_argument('receipt', type=pathlib.Path); x.add_argument('--store', type=pathlib.Path, required=True)
    x = sub.add_parser('record-audit'); x.add_argument('id'); x.add_argument('patch', type=pathlib.Path); x.add_argument('receipt', type=pathlib.Path); x.add_argument('unresolved', type=pathlib.Path); x.add_argument('--store', type=pathlib.Path, required=True)
    x = sub.add_parser('record-adjudication'); x.add_argument('id'); x.add_argument('decision', type=pathlib.Path); x.add_argument('--store', type=pathlib.Path, required=True)
    x = sub.add_parser('finalize'); x.add_argument('id'); x.add_argument('--store', type=pathlib.Path, required=True)
    x = sub.add_parser('status'); x.add_argument('--store', type=pathlib.Path, required=True)
    x = sub.add_parser('annotation-schema'); x.add_argument('id')
    x = sub.add_parser('audit-schema'); x.add_argument('--prediction', type=pathlib.Path)
    x = sub.add_parser('schema'); x.add_argument('--production', action='store_true', default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.command in ('audit-schema', 'annotation-schema'):
        audit_source = Sources(args.packets, args.raw, args.vocabulary)
        audit_source.production = True
        if not args.vocabulary:
            audit_source.vocab = copy.deepcopy(PRODUCTION_FAMILIES)
        emit(bound_annotation_schema(args.id, audit_source) if args.command == 'annotation-schema' else audit_output_schema(load(args.prediction) if args.prediction else None, audit_source))
        return
    if args.command == 'schema' and args.production:
        schema_source = Sources(args.packets, args.raw, args.vocabulary)
        if not args.vocabulary:
            schema_source.vocab = copy.deepcopy(PRODUCTION_FAMILIES)
        emit(production_schema(schema_source.vocab, schema_source.extra_enums))
        return
    if args.command == 'schema':
        emit({'version': 3, 'enums': {'purpose': PURPOSE, 'exec': EXEC, 'feedback': FEEDBACK, 'criterion': CRITERIA, 'action': ACTIONS, 'link_kind': KINDS, 'link_order': ORDERS, 'state': STATES}, 'families': FAMILIES,
              'unfrozen_fields': ['file.role', 'choice.relation', 'additional recoding families'],
              'note': 'Contract leaves these fields open; freeze through --vocabulary before production. Template nulls are invalid until reviewed.'})
        return
    source = Sources(args.packets, args.raw, args.vocabulary)
    source.production = args.production or args.command in ('validate-audit', 'record-audit-output', 'audit-errors')
    if source.production and not args.vocabulary:
        source.vocab = copy.deepcopy(PRODUCTION_FAMILIES)
    if args.command == 'read':
        if args.production:
            require(len(args.ids) == 1, 'production source reader permits exactly one case per context')
        source.collect(args.ids)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            for identifier in args.ids:
                f.write(source.full_text(identifier, raw_events=args.raw_events))
        emit({'output': str(args.output), 'cases': len(args.ids), 'complete_delimiters': True, 'truncated': False})
    elif args.command in ('template', 'manifest'):
        obj = template(args.id, source) if args.command == 'template' else manifest(source, args.batch_size)
        immutable(args.output, obj)
        emit({'output': str(args.output)})
    elif args.command in ('validate', 'features', 'coverage', 'canonicalize', 'errors'):
        rr = records(args.annotations)
        require(len({r['id'] for r in rr}) == len(rr), 'duplicate case annotations')
        source.collect([r['id'] for r in rr])
        results = []
        for record in rr:
            try:
                if args.command == 'errors':
                    results.append(collect_record_errors(record, source))
                elif args.command == 'canonicalize':
                    annotation, reconciliation = canonicalize_derived(record, source)
                    results.append({'annotation': annotation, 'reconciliation': reconciliation, 'validation': validation_snapshot(annotation, source)})
                else:
                    results.append(validate(record, source) if args.command == 'validate' else (coverage_check(record, source) if args.command == 'coverage' else features(record, source)))
            except (Invalid, KeyError, TypeError, ValueError) as error:
                results.append({'valid': False, 'id': record.get('id'), 'error': str(error), 'semantic_correctness_established': False})
        emit(results)
        if any(result.get('valid') is False for result in results):
            raise SystemExit(2)
    elif args.command == 'diff':
        aa, bb = records(args.A), records(args.B)
        source.collect([r['id'] for r in aa + bb])
        a_by = {source.by_id[r['id']]['case_id']: r for r in aa}
        b_by = {source.by_id[r['id']]['case_id']: r for r in bb}
        require(len(a_by) == len(aa) and len(b_by) == len(bb), 'duplicate cases in comparison')
        comparisons = []
        for case in sorted(set(a_by) | set(b_by)):
            if case not in a_by or case not in b_by:
                comparisons.append({'source_case_id': case, 'missing_A': case not in a_by, 'missing_B': case not in b_by})
                continue
            a, b = a_by[case], b_by[case]
            require(a['hash'] == b['hash'], 'different source hashes cannot be compared')
            item = differences(a, b)
            item['source_case_id'] = case
            for label, record in [('A', a), ('B', b)]:
                try:
                    item['validation_' + label] = validate(record, source)
                except (Invalid, KeyError, TypeError, ValueError) as error:
                    item['validation_' + label] = {'valid': False, 'error': str(error)}
            for label, record in [('A', a), ('B', b)]:
                item['coverage_' + label] = coverage_check(record, source)
                derived = derive_feedback(record)
                item['feedback_discrepancies_' + label] = [{'id': row['id'], 'provided': row['feedback'], 'derived': derived[row['id']]} for row in record['commands'] if row['feedback'] != derived[row['id']]]
                item['review_flags_' + label] = review_flags(record, source)
            comparisons.append(item)
        emit({'A_path': str(args.A), 'A_sha256': sha(args.A.read_bytes()), 'B_path': str(args.B), 'B_sha256': sha(args.B.read_bytes()),
              'exposure_note': args.exposure_note, 'comparisons': comparisons, 'semantic_correctness_established': False,
              'interpretation': 'Structural discrepancies, including invalid records, are preserved; no agreement-as-truth claim.'})
    elif args.command == 'receipt-template':
        p, (_, lines, _) = source.get(args.id)
        obj = {'source_sha256': p['eventstream_sha256'], 'full_source_read': None, 'predictions_visible': None,
               'negative_items_checked': None, 'prediction_sha256': None, 'context_case_count': None,
               'covered_items': {g: [i['item_id'] for i in p['items'] if GROUPS[i['type']] == g] for g in ('commands', 'messages', 'files')},
               'event_line_ranges': [[1, len(lines)]], 'reviewer': None, 'model': None, 'effort': None, 'tier': None,
               'prompt_sha256': None, 'token_usage': {'unavailable': None}, 'cost': {'unavailable': None}}
        immutable(args.output, obj)
        emit({'output': str(args.output), 'valid_receipt': False, 'note': 'Mechanical expected coverage only; explicit attestation and actual run metadata remain required.'})
    elif args.command == 'record-a':
        raw_text = args.annotation.read_text()
        a, receipt = json.loads(raw_text), load(args.receipt)
        check = validation_snapshot(a, source)
        is_draft = not check['valid']
        if is_draft:
            require(args.allow_invalid_draft, 'invalid first pass requires --allow-invalid-draft: ' + check['error'])
            coverage = draft_coverage_check(a, source, allow_coverage_errors=args.allow_coverage_errors)
        else:
            coverage = coverage_check(a, source)
        receipt_check(receipt, a, source, draft=is_draft)
        artifact = {'schema': 'card-pass-A-v3', 'state': 'draft_invalid' if is_draft else 'validated', 'draft_coverage_errors_allowed': bool(is_draft and args.allow_coverage_errors),
                    'annotation': a, 'annotation_sha256': sha(canonical(a)), 'receipt': receipt,
                    'validation': check, 'coverage': coverage, 'raw_coverage_and_evidence': coverage_check(a, source), 'raw_annotation_text': raw_text,
                    'raw_annotation_sha256': sha(args.annotation.read_bytes())}
        target = safe_dir(args.store, a, source) / 'pass-A.json'
        immutable(target, artifact); emit({'saved': str(target), 'state': artifact['state'], 'valid': not is_draft, 'sha256': sha(canonical(artifact))})
    elif args.command == 'audit-errors':
        emit(collect_audit_errors(load(args.prediction), load(args.audit_output), source))
    elif args.command == 'validate-audit':
        emit(validate_audit_output(load(args.prediction), load(args.audit_output), source))
    elif args.command == 'record-audit-output':
        directory = safe_dir(args.store, {'id': args.id}, source)
        before = load(directory / 'pass-A.json')
        document, receipt = load(args.audit_output), load(args.receipt)
        audit_check = validate_audit_output(before['annotation'], document, source)
        require(receipt.get('prediction_sha256') == before['annotation_sha256'], 'audit receipt must bind exact saved pass A')
        after, canonicalization = canonicalize_derived(document['annotation'], source)
        receipt_check(receipt, after, source, audit=True)
        patch = whole_record_patch(before['annotation'], after)
        require(patch_apply(before['annotation'], patch, source=source) == after, 'full audit reconstruction mismatch')
        artifact = {'schema': 'card-source-audit-v3', 'before': before['annotation'], 'prediction_sha256': before['annotation_sha256'],
                    'patch': patch, 'after': after, 'annotation_sha256': sha(canonical(after)), 'receipt': receipt,
                    'audit_output': document, 'audit_validation': audit_check, 'canonicalization': canonicalization,
                    'unresolved_adjudication': document['unresolved_adjudication'], 'discrepancies': differences(before['annotation'], after)}
        immutable(directory / 'source-audit.json', artifact)
        emit({'saved': str(directory / 'source-audit.json'), 'valid_audited_annotation': True, 'unresolved_count': len(document['unresolved_adjudication'])})
    elif args.command == 'record-audit':
        require(not source.production, 'production requires record-audit-output with explicit source-backed decision for every item')
        directory = safe_dir(args.store, {'id': args.id}, source)
        before = load(directory / 'pass-A.json')
        patch, receipt, unresolved = load(args.patch), load(args.receipt), load(args.unresolved)
        require(receipt.get('prediction_sha256') == before['annotation_sha256'], 'audit must bind exact pass A predictions')
        after = patch_apply(before['annotation'], patch)
        validate(after, source); receipt_check(receipt, after, source, audit=True); unresolved_list(unresolved)
        artifact = {'schema': 'card-source-audit-v3', 'before': before['annotation'], 'prediction_sha256': before['annotation_sha256'],
                    'patch': patch, 'after': after, 'annotation_sha256': sha(canonical(after)), 'receipt': receipt,
                    'unresolved_adjudication': unresolved, 'discrepancies': differences(before['annotation'], after)}
        target = directory / 'source-audit.json'
        immutable(target, artifact); emit({'saved': str(target)})
    elif args.command == 'record-adjudication':
        directory = safe_dir(args.store, {'id': args.id}, source)
        audit = load(directory / 'source-audit.json')
        require(not (directory / 'final.json').exists(), 'cannot adjudicate after immutable final; use new versioned store')
        require(not (directory / 'adjudication.json').exists(), 'adjudication already stored; use new versioned store')
        decision = load(args.decision)
        pending = {r['id'] for r in audit['unresolved_adjudication']}
        require(set(decision.get('resolved_ids', [])) <= pending, 'adjudication references unknown dispute')
        unresolved_list(decision.get('unresolved_adjudication'))
        require(set(decision['resolved_ids']).isdisjoint({x['id'] for x in decision['unresolved_adjudication']}), 'dispute both resolved and unresolved')
        require(set(decision['resolved_ids']) | {x['id'] for x in decision['unresolved_adjudication']} == pending, 'adjudication silently omits disputes')
        require(decision.get('before_sha256') == audit['annotation_sha256'], 'adjudication before hash mismatch')
        after = patch_apply(audit['after'], decision.get('patch'))
        validate(after, source)
        receipt_check(decision.get('receipt'), after, source, audit=True)
        immutable(directory / 'adjudication.json', {'schema': 'card-adjudication-v3', 'decision': decision, 'after': after, 'annotation_sha256': sha(canonical(after))})
        emit({'saved': str(directory / 'adjudication.json')})
    elif args.command == 'finalize':
        directory = safe_dir(args.store, {'id': args.id}, source)
        before = load(directory / 'pass-A.json'); audit = load(directory / 'source-audit.json')
        a = audit['after']; unresolved = audit['unresolved_adjudication']
        if source.production:
            require('audit_output' in audit, 'production finalization requires explicit complete item audit output')
            validate_audit_output(before['annotation'], audit['audit_output'], source, deduplicate_sets=audit.get('canonicalization', {}).get('set_deduplication_enabled', False))
        adjudication = directory / 'adjudication.json'
        refs = {'pass_A_sha256': sha(canonical(before)), 'source_audit_sha256': sha(canonical(audit))}
        if adjudication.exists():
            adj = load(adjudication); a = adj['after']; unresolved = adj['decision']['unresolved_adjudication']
            refs['adjudication_sha256'] = sha(canonical(adj))
        validate(a, source); unresolved_list(unresolved)
        immutable(directory / 'final.json', {'schema': 'card-final-v3', 'annotation': a, 'annotation_sha256': sha(canonical(a)),
                                             'unresolved_adjudication': unresolved, 'status': 'unresolved' if unresolved else 'audited', 'artifacts': refs})
        emit({'saved': str(directory / 'final.json'), 'unresolved_count': len(unresolved)})
    elif args.command == 'status':
        # One stream read for all present cases, rather than per-case scans.
        present = [r['case_id'] for r in source.index if (args.store / (r.get('coding_id') or r['case_id'])).exists()]
        source.collect(present)
        emit(status(source, args.store))


if __name__ == '__main__':
    try:
        main()
    except (Invalid, KeyError, TypeError, ValueError, FileExistsError, FileNotFoundError) as e:
        print(json.dumps({'valid': False, 'error': str(e)}), file=sys.stderr)
        sys.exit(2)
