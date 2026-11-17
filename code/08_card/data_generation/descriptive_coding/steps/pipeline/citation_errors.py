"""All-error collection helpers; no source labels or quotes are changed.

Imported by card_review only after its definitions exist. This separate module
keeps repair reporting separate from fail-fast final acceptance.
"""
import copy


def schema_errors(value, schema, path='$'):
    errors = []
    types = schema.get('type')
    types = [types] if isinstance(types, str) else types
    checks = {'object': lambda x: isinstance(x, dict), 'array': lambda x: isinstance(x, list),
              'string': lambda x: isinstance(x, str), 'integer': lambda x: type(x) is int,
              'boolean': lambda x: type(x) is bool, 'null': lambda x: x is None}
    if types and not any(checks[t](value) for t in types):
        return [{'path': path, 'kind': 'schema', 'error': 'wrong JSON type'}]
    if 'enum' in schema and value not in schema['enum']:
        errors.append({'path': path, 'kind': 'schema', 'error': 'outside allowed enum'})
    if isinstance(value, dict):
        props = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                errors.append({'path': path + '.' + key, 'kind': 'schema', 'error': 'missing required field'})
        if schema.get('additionalProperties') is False:
            for key in sorted(set(value) - set(props)):
                errors.append({'path': path + '.' + key, 'kind': 'schema', 'error': 'unexpected property'})
        for key in props:
            if key in value:
                errors.extend(schema_errors(value[key], props[key], path + '.' + key))
    elif isinstance(value, list):
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', len(value)):
            errors.append({'path': path, 'kind': 'schema', 'error': 'wrong array length'})
        for n, row in enumerate(value):
            errors.extend(schema_errors(row, schema.get('items', {}), path + '[' + str(n) + ']'))
    elif isinstance(value, str) and len(value) < schema.get('minLength', 0):
        errors.append({'path': path, 'kind': 'schema', 'error': 'empty string'})
    elif type(value) is int and value < schema.get('minimum', value):
        errors.append({'path': path, 'kind': 'schema', 'error': 'below minimum'})
    return errors


def record_errors(cr, record, store, canonicalize=True, deduplicate_sets=False):
    errors, matches = [], []
    candidate = copy.deepcopy(record)
    reconciliation = None
    caught = (cr.Invalid, KeyError, TypeError, ValueError, AttributeError, StopIteration, IndexError)
    def check(path, kind, function):
        try:
            result = function()
            if isinstance(result, dict) and result.get('match_method'):
                matches.append(dict(result, path=path))
            return result
        except caught as error:
            errors.append({'path': path, 'kind': kind, 'error': str(error)})
    if canonicalize:
        result = check('$', 'derived_metadata', lambda: cr.canonicalize_record(record, store, deduplicate_sets=deduplicate_sets))
        if result:
            candidate, reconciliation = result
    if store.production:
        schema = check('$', 'source_binding', lambda: cr.bound_annotation_schema(candidate['id'], store))
        if schema:
            errors.extend(schema_errors(candidate, schema))
    packet = check('$.id', 'source_binding', lambda: store.get(candidate['id'])[0])
    roster = store.item_roster(candidate['id']) if packet else {}
    for group in ('commands', 'messages', 'files'):
        rows = candidate.get(group, [])
        if not isinstance(rows, list):
            continue
        ids = [r.get('id') for r in rows if isinstance(r, dict)]
        expected = {i for i, item in roster.items() if cr.GROUPS[item['type']] == group}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            errors.append({'path': '$.' + group, 'kind': 'coverage', 'error': 'missing, extra, or duplicated item IDs'})
        for n, row in enumerate(rows):
            path = '$.' + group + '[' + str(n) + ']'
            if not isinstance(row, dict):
                continue
            check(path + '.ev', 'citation_lines', lambda row=row: store.evidence(candidate['id'], row, check_quote=False))
            if group in ('commands', 'messages') and (store.production or 'quote' in row) or group == 'files' and 'quote' in row:
                check(path + '.quote', 'own_item_quote', lambda row=row, group=group: cr.item_quote_check(candidate, row, store, group))
            if row.get('id') in roster:
                item = roster[row['id']]
                pair = item['occurrence_lines']
                if not any(line in row.get('ev', []) for line in pair if line is not None):
                    errors.append({'path': path + '.ev', 'kind': 'citation_lines', 'error': 'no cited event belongs to this item'})
            if group == 'commands':
                if row.get('main_purpose') not in (None, 'mixed') and row.get('main_purpose') not in row.get('purpose', []):
                    errors.append({'path': path + '.main_purpose', 'kind': 'consistency', 'error': 'dominant purpose absent from purpose list'})
                for family in row.get('families', []) or []:
                    if isinstance(family, str) and ':' in family and family.split(':', 1)[0] not in row.get('dim', []):
                        errors.append({'path': path + '.families', 'kind': 'consistency', 'error': 'family dimension absent from dim: ' + family})
                if row.get('exec') in ('none', 'definition_only', 'attempt_failed') and row.get('families'):
                    errors.append({'path': path + '.families', 'kind': 'consistency', 'error': 'unexecuted/failed command cannot contribute evaluated families'})
            if group == 'messages' and set(row.get('ref', []) or []) - set(roster):
                errors.append({'path': path + '.ref', 'kind': 'consistency', 'error': 'dangling item reference'})
    for n, link in enumerate(candidate.get('links', []) or []):
        path = '$.links[' + str(n) + ']'
        if not isinstance(link, dict):
            continue
        check(path + '.from', 'citation_lines', lambda link=link: store.evidence(candidate['id'], link, 'from', check_quote=False))
        check(path + '.to', 'citation_lines', lambda link=link: store.evidence(candidate['id'], link, 'to', check_quote=False))
        check(path + '.quote', 'cited_event_quote', lambda link=link: store.evidence(candidate['id'], {'ev': sorted(set(link.get('from', []) + link.get('to', []))), 'quote': link.get('quote')}))
    portfolio = candidate.get('portfolio', {})
    if isinstance(portfolio, dict):
        for dim, rows in portfolio.items():
            if isinstance(rows, list):
                for n, row in enumerate(rows):
                    check('$.portfolio.' + dim + '[' + str(n) + '].ev', 'citation_lines', lambda row=row: store.evidence(candidate['id'], row))
    if isinstance(candidate.get('choice'), dict):
        choice = candidate['choice']
        check('$.choice.ev', 'citation_lines', lambda: store.evidence(candidate['id'], choice, require_nonempty=any(choice.get(k) for k in ('stated', 'implemented', 'emphasized'))))
    # Preserve any remaining consistency error from the strict acceptance gate.
    try:
        cr.validate_record(candidate, store)
    except caught as error:
        if not any(e['error'] == str(error) for e in errors):
            errors.append({'path': '$', 'kind': 'acceptance_gate', 'error': str(error)})
    return {'valid': not errors, 'errors': errors, 'quote_matches': matches,
            'canonicalization': reconciliation, 'canonical_annotation_sha256': cr.sha(cr.canonical(candidate)),
            'semantic_correctness_established': False}


def audit_errors(cr, prediction, document, store, deduplicate_sets=False):
    errors, matches, verdicts = [], [], []
    caught = (cr.Invalid, KeyError, TypeError, ValueError, AttributeError, StopIteration, IndexError)
    try:
        schema = cr.audit_output_schema(prediction, store)
        errors.extend(schema_errors(document, schema))
    except caught as error:
        errors.append({'path': '$', 'kind': 'source_binding', 'error': str(error)})
    report = record_errors(cr, document.get('annotation', {}), store, deduplicate_sets=deduplicate_sets)
    errors.extend(dict(e, path='$.annotation' + e['path'][1:]) for e in report['errors'])
    matches.extend(dict(m, path='$.annotation' + m['path'][1:]) for m in report['quote_matches'])
    annotation = document.get('annotation', {})
    try:
        packet = store.get(annotation['id'])[0]
        roster = store.item_roster(annotation['id'])
    except caught:
        roster = {}
    pending = {r.get('id') for r in document.get('unresolved_adjudication', []) if isinstance(r, dict)}
    semantic = {'commands': ['purpose', 'main_purpose', 'dim', 'exec', 'families'], 'messages': ['criterion', 'action', 'ref'], 'files': ['role', 'choice']}
    for group in ('commands', 'messages', 'files'):
        before = {}
        for raw_row in prediction.get(group, []): before.setdefault(raw_row['id'], []).append(raw_row)
        after = {r['id']: r for r in annotation.get(group, [])}
        rows = document.get('item_audit', {}).get(group, [])
        ids = [row.get('id') for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != {key for key, item in roster.items() if cr.GROUPS[item['type']] == group}:
            errors.append({'path': '$.item_audit.' + group, 'kind': 'audit_coverage', 'error': 'missing, extra, or duplicated audit IDs'})
        for n, row in enumerate(rows):
            path = '$.item_audit.' + group + '[' + str(n) + ']'
            try:
                store.evidence(annotation['id'], row, check_quote=False)
            except caught as error:
                errors.append({'path': path + '.ev', 'kind': 'citation_lines', 'error': str(error)})
            try:
                match = cr.item_quote_check(annotation, row, store, group)
                matches.append(dict(match, path=path + '.quote'))
            except caught as error:
                errors.append({'path': path + '.quote', 'kind': 'own_item_quote', 'error': str(error)})
            if not isinstance(row.get('reason'), str) or not row['reason'].strip():
                errors.append({'path': path + '.reason', 'kind': 'audit_reason', 'error': 'source-specific reason must be nonempty'})
            key = row.get('id')
            if key in roster:
                own = roster[key]
                expected_lines = own['occurrence_lines']
                if not any(line in row.get('ev', []) for line in expected_lines if line is not None):
                    errors.append({'path': path + '.ev', 'kind': 'citation_lines', 'error': 'no cited event belongs to this audit item'})
            if key in after:
                raw_rows = before.get(key, [])
                prior = raw_rows[0] if len(raw_rows) == 1 else (raw_rows or None)
                verdicts.append(dict(cr.derive_audit_verdict(prior, after[key], group, row.get('decision')), id=key, group=group))
            if row.get('decision') == 'unresolved' and key not in pending:
                errors.append({'path': path + '.decision', 'kind': 'audit_decision', 'error': 'unresolved item missing from unresolved_adjudication'})
    try:
        cr.validate_audit(prediction, document, store, deduplicate_sets=deduplicate_sets)
    except caught as error:
        if not any(e['error'] == str(error) for e in errors):
            errors.append({'path': '$', 'kind': 'acceptance_gate', 'error': str(error)})
    return {'valid': not errors, 'errors': errors, 'quote_matches': matches, 'audit_verdicts': verdicts, 'annotation_reconciliation': report['canonicalization'],
            'semantic_correctness_established': False}
