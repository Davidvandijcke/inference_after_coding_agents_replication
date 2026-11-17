"""Frozen, text-inert Card process features. No external labels are read here.

Only audited semantic labels and explicitly reviewed typed sidecars are consumed.
No command/source text is executed or classified. See README.md for contracts.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

VERSION = 'card-recorded-search-display-v2'
GROUPS = ('fte', 'fulltime', 'headcount', 'parttime', 'manager', 'other_employment')
GROUP_LABELS = ('Full-time-equivalent employment', 'Full-time employees', 'Headcount',
                'Part-time employees', 'Managers', 'Transformed / other employment')
OUTCOME_MAP = {'fte_change': 'fte', 'fte_level': 'fte', 'fulltime_change': 'fulltime',
               'fulltime_level': 'fulltime', 'headcount_change': 'headcount',
               'headcount_level': 'headcount', 'parttime': 'parttime', 'manager': 'manager',
               'transformed_employment': 'other_employment', 'other': 'other_employment'}
MOTIFS = ('multiple_outcome_groups_one_command', 'new_outcome_group_after_result',
          'scientific_evaluation_after_primary', 'verification_after_last_primary')
MOTIF_LABELS = ('Multiple outcomes in one command', 'New outcome after an earlier result',
                'Scientific evaluation after primary declaration', 'Verification after last primary declaration')
PURPOSES = ('prepare/inspect', 'evaluate', 'interpret/choose', 'verify', 'debug', 'report', 'mixed/other')
PURPOSE_MAP = {'setup': PURPOSES[0], 'inspect': PURPOSES[0], 'construct': PURPOSES[0],
               'evaluate': PURPOSES[1], 'interpret': PURPOSES[2], 'choose': PURPOSES[2],
               'revise': PURPOSES[2], 'verify': PURPOSES[3], 'debug': PURPOSES[4],
               'report': PURPOSES[5], 'mixed': PURPOSES[6], 'other': PURPOSES[6], 'unknown': PURPOSES[6]}
CRITERIA = ('design', 'measurement', 'direction', 'magnitude', 'precision', 'p_significance',
            'inference_convention', 'technical', 'other', 'unclear')
STATES = ('positive', 'negative', 'unresolved', 'no_declaration_observed')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def cohort_digest(cases):
    return digest(sorted(cases, key=lambda c: c['id']))


DEFINITIONS = {
    'version': VERSION, 'outcome_groups': list(zip(GROUPS, GROUP_LABELS)),
    'outcome_map': OUTCOME_MAP, 'motifs': list(zip(MOTIFS, MOTIF_LABELS)),
    'purpose_map': PURPOSE_MAP, 'criteria': CRITERIA,
    'ordering': 'earlier command completion < later command start; declaration evidence line < later start',
    'position': '(one-based rank - 0.5) / total recorded commands; all commands retained',
    'activity_weight': 'each case contributes 1/its recorded command count across its commands',
    'eligibility': 'audited evaluated/partial_results/mixed + evaluate purpose + substantive family; verification/rerun/descriptive-only excluded',
    'novelty': 'new broad outcome absent from earlier-started commands; overlapping or unknown earlier evidence blocks confirmation',
    'declaration': 'audited message primary_adopt/primary_revise at source line; auditors distinguish planned selection from actual adoption',
    'criteria': 'explicit message criterion presence only; no criterion-to-action causal or private-reasoning claim',
    'unknown_policy': 'no complete audited source coverage => no definitive negative; pending claims globally conservative unless exact-final-hash Root scope clearance enumerates excluded claims and affected items/features; raw disputes retained',
    'groups_are_marginal': True, 'original_in_replay_denominator': False,
    'uncertainty': 'coding range P/N to (P+U)/N; no inferential error bars',
    'example_selection': 'minimum sha256(display-example-v1|neutral_id), independent of external labels',
}
DEFINITIONS_SHA256 = digest(DEFINITIONS)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def outcome_groups(tokens):
    """Exact frozen enum lookup. Unknown tokens remain explicit, never regex-coded."""
    known, unknown = set(), []
    for token in tokens:
        if token.startswith('O:'):
            group = OUTCOME_MAP.get(token[2:])
            if group is None:
                unknown.append(token)
            else:
                known.add(group)
    return known, sorted(unknown)


def indicator(positive=False, uncertain=False, witnesses=(), reason=None, no_declaration=False):
    state = ('positive' if positive else 'unresolved' if uncertain else
             'no_declaration_observed' if no_declaration else 'negative')
    return {'state': state, 'witnesses': list(witnesses), 'reason': reason}


def _binding(sidecar, identifier, source_hash, annotation_hash, name, annotation_required=True):
    if sidecar is None:
        return
    check(sidecar.get('id') == identifier and sidecar.get('source_sha256') == source_hash,
          name + ': identity/source mismatch')
    if annotation_required or 'annotation_sha256' in sidecar:
        check(sidecar.get('annotation_sha256') == annotation_hash, name + ': annotation mismatch')
    check(sidecar.get('review_ref'), name + ': reviewed provenance required')


def _lines(lines, n, name):
    check(isinstance(lines, list) and lines and all(type(x) is int and 1 <= x <= n for x in lines),
          name + ': nonempty source evidence lines required')


def derive_case(final, packet, reviewed_choice=None, evidence=None, scope_clearance=None):
    """Derive one neutral case before any external stratum/match join.

    final is an immutable card-final-v3 artifact or None for a missing final.
    packet is hash-bound metadata, not a mutable lookup of scientific outcomes.
    reviewed_choice and evidence are typed, hash-bound reviewed adapters.
    """
    identifier, source_hash = packet['coding_id'], packet['eventstream_sha256']
    check(packet.get('packet_sha256') and type(packet.get('is_original')) is bool,
          'immutable packet hash and original flag required')
    nlines = packet['line_count']
    check(type(nlines) is int and nlines > 0, 'source line_count required')
    result = {'id': identifier, 'version': VERSION, 'definitions_sha256': DEFINITIONS_SHA256,
              'source_sha256': source_hash, 'packet_sha256': packet['packet_sha256'],
              'is_original': packet['is_original'], 'annotation_sha256': None,
              'flags': [], 'activity': [], 'criteria': {}, 'family_portfolio': {},
              'evaluated': {}, 'declared': {}, 'motifs': {}, 'declarations': [],
              'commands': [], 'event_items': [], 'raw_choice': None}
    if final is None:
        result['flags'] = ['missing_final_annotation']
        for field, names in (('evaluated', GROUPS), ('declared', GROUPS), ('motifs', MOTIFS), ('criteria', CRITERIA)):
            result[field] = {x: indicator(uncertain=True, reason='missing final annotation') for x in names}
        result['declaration_status'] = 'unresolved'
        result['breadth'] = {'min': 0, 'max': 5}
        return result
    check(final.get('schema') == 'card-final-v3', 'finalized artifact required; draft A/B disallowed')
    r = final['annotation']
    ahash = digest(r)
    check(r['id'] == identifier and r['hash'] == source_hash, 'annotation/source mismatch')
    check(final['annotation_sha256'] == ahash, 'final annotation hash mismatch')
    check(final.get('artifacts', {}).get('pass_A_sha256') and final['artifacts'].get('source_audit_sha256'),
          'final must bind pass A and complete source audit')
    validation = packet.get('source_validation', {})
    item_scoped = (validation.get('mode') == 'root_item_scoped_source_validation' and
                   validation.get('approved_by') == 'Root' and validation.get('scope_policy_sha256'))
    check((validation.get('valid') is True or item_scoped) and validation.get('annotation_sha256') == ahash,
          'upstream full-source or Root item-scoped validation receipt must bind final annotation')
    check(validation.get('source_sha256') == source_hash, 'validation receipt source mismatch')
    check(validation.get('final_sha256') == digest(final), 'source validation must bind exact final artifact')
    check(validation.get('review_ref'), 'source-validation receipt provenance required')
    result['annotation_sha256'] = ahash
    result['final_sha256'] = digest(final)
    result['source_validation'] = validation
    result['raw_choice'] = r['choice']
    cov = r['coverage']
    pending = final.get('unresolved_adjudication', [])
    check(final.get('status') in ('audited', 'unresolved'), 'invalid final artifact status')
    scoped_items, scoped_features = set(), set()
    unresolved = bool(pending) or final.get('status') != 'audited'
    if scope_clearance is not None:
        _binding(scope_clearance, identifier, source_hash, ahash, 'reporting scope clearance')
        check(scope_clearance.get('approved_by') == 'Root' and scope_clearance.get('final_sha256') == digest(final),
              'Root reporting scope must bind exact final artifact')
        decisions = scope_clearance.get('pending_claims', [])
        check(len({d['id'] for d in decisions}) == len(decisions) and
              {d['id'] for d in decisions} == {d['id'] for d in pending}, 'scope must disposition every pending claim exactly once')
        roster_ids = {row['id'] for g in ('commands', 'messages', 'files') for row in r[g]}
        valid_features = {field + '.' + name for field, names in (
            ('evaluated', GROUPS), ('declared', GROUPS), ('motifs', MOTIFS), ('criteria', CRITERIA)) for name in names}
        for decision in decisions:
            check(decision.get('scope') in ('excluded', 'affected') and decision.get('reason'), 'explicit scope disposition/reason required')
            items, names = set(decision.get('affected_item_ids', [])), set(decision.get('affected_features', []))
            check(items <= roster_ids and names <= valid_features, 'scope contains unknown items/features')
            check((not items and not names) if decision['scope'] == 'excluded' else bool(items or names),
                  'excluded claims have no reporting effects; affected claims must identify their effects')
            scoped_items.update(items); scoped_features.update(names)
        # Clearance resolves reporting scope only; it never resolves or erases disputes.
        unresolved = False
    result['reporting_scope_clearance'] = scope_clearance
    result['scope_affected_items'] = sorted(scoped_items)
    result['scope_affected_features'] = sorted(scoped_features)
    unknown_items = set(cov['unknown_items']) | scoped_items
    if item_scoped:
        check(set(validation.get('invalid_item_ids', [])) <= unknown_items,
              'every source-invalid item must remain unknown under Root scope')
    base_complete = cov['complete'] and not cov['unavailable'] and not unresolved
    command_ids = {c['id'] for c in r['commands']}
    message_ids = {m['id'] for m in r['messages']}
    complete = base_complete and not (unknown_items & command_ids)
    messages_complete = base_complete and not (unknown_items & message_ids)
    result['coverage'] = cov
    result['unresolved_adjudication'] = final.get('unresolved_adjudication', [])
    if not complete:
        result['flags'].append('incomplete_or_unresolved_annotation')
    external_choice = (reviewed_choice or {}).get('source_kind') == 'external_reviewed_declaration'
    _binding(reviewed_choice, identifier, source_hash, ahash, 'reviewed choice', annotation_required=not external_choice)
    if external_choice:
        artifact = reviewed_choice.get('source_artifact', {})
        for label, value in (('formal declaration', reviewed_choice.get('declaration_sha256')),
                             ('source artifact', artifact.get('sha256')),
                             ('source record', artifact.get('record_sha256'))):
            check(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                  label + ': exact SHA-256 provenance required')
        check(artifact.get('path') and type(artifact.get('line')) is int and artifact['line'] >= 1,
              'external declaration source artifact path/line required')
    _binding(evidence, identifier, source_hash, ahash, 'component/declaration evidence')
    result['reporting_evidence'] = evidence
    overrides = (evidence or {}).get('commands', {})
    command_ids = {c['id'] for c in r['commands']}
    check(set(overrides) <= command_ids, 'component sidecar contains unknown commands')
    commands = sorted(r['commands'], key=lambda c: (c['at'][0], c['id']))
    for rank, c in enumerate(commands, 1):
        start, end = c['at']
        check(type(start) is int and 1 <= start <= nlines and
              (end is None or type(end) is int and start <= end <= nlines), 'invalid command span')
        if c['id'] not in unknown_items:
            _lines(c['ev'], nlines, c['id'])
        groups, unknown = outcome_groups(c['families'])
        substantive = any(x.split(':', 1)[0] in ('O', 'S', 'M', 'W', 'T') for x in c['families'])
        positive = c['exec'] in ('evaluated', 'partial_results', 'mixed') and 'evaluate' in c['purpose'] and substantive
        eligible = 'confirmed' if positive else 'unresolved' if c['exec'] == 'unknown' else 'excluded'
        if c['exec'] in ('evaluated', 'partial_results', 'mixed') and 'evaluate' in c['purpose'] and not c['families']:
            eligible = 'unresolved'
        reason = 'audited execution/purpose/families'
        verify = 'confirmed' if c['exec'] == 'verification_only' and 'verify' in c['purpose'] else (
                 'unresolved' if c['exec'] in ('mixed', 'unknown') and 'verify' in c['purpose'] else 'excluded')
        override = overrides.get(c['id'])
        if override:
            _lines(override.get('ev'), nlines, c['id'] + ' override')
            check(override.get('review_ref'), 'component review reference required')
            check(override.get('evaluation') in ('confirmed', 'excluded', 'unresolved'), 'invalid evaluation disposition')
            check(override.get('verification') in ('confirmed', 'excluded', 'unresolved'), 'invalid verification disposition')
            eligible, verify, reason = override['evaluation'], override['verification'], override['review_ref']
            if eligible == 'confirmed':
                check(c['exec'] in ('evaluated', 'partial_results', 'mixed') and substantive,
                      'confirmed component requires actual scientific results/families')
        # An unmatched start does not establish completed evaluation/verification.
        if end is None:
            if eligible == 'confirmed': eligible = 'unresolved'
            if verify == 'confirmed': verify = 'unresolved'
        if unknown:
            eligible = 'unresolved'
            result['flags'].append('unmapped_outcome_tokens:' + c['id'])
        if unresolved or c['id'] in unknown_items:
            if eligible == 'confirmed': eligible = 'unresolved'
            if verify == 'confirmed': verify = 'unresolved'
        row = {'id': c['id'], 'start': start, 'end': end, 'evaluation': eligible,
               'verification': verify, 'groups': sorted(groups), 'families': c['families'],
               'unknown_tokens': unknown, 'ev': c['ev'], 'quote': c.get('quote'),
               'raw_quote_ref': c.get('quote_ref'), 'basis': reason}
        result['commands'].append(row)
        result['activity'].append({'id': c['id'], 'rank': rank, 'total_commands': len(commands),
                                   'position': (rank - .5) / len(commands),
                                   'case_weight': 1 / len(commands),
                                   'state': PURPOSES[-1] if c['id'] in unknown_items else PURPOSE_MAP.get(c['main_purpose'], PURPOSES[-1]),
                                   'raw_main_purpose': c['main_purpose'], 'ev': c['ev'],
                                   'annotation_uncertain': unresolved or c['id'] in unknown_items})
    result['event_items'] = [{'id': x['id'], 'kind': group, 'at': x['at'], 'ev': x['ev']}
                             for group in ('commands', 'messages', 'files') for x in r[group]]
    cc = result['commands']
    result['overlapping_commands'] = [
        {'first': a['id'], 'second': b['id'], 'first_span': [a['start'], a['end']],
         'second_span': [b['start'], b['end']]}
        for i, a in enumerate(cc) for b in cc[i + 1:]
        if a['end'] is None or a['end'] >= b['start']]
    evals = [c for c in cc if c['evaluation'] == 'confirmed']
    potential = [c for c in cc if c['evaluation'] == 'unresolved']
    for g in GROUPS:
        witnesses = [c for c in evals if g in c['groups']]
        result['evaluated'][g] = indicator(bool(witnesses), not complete or bool(potential), witnesses)
    # Novelty checks every earlier-started command, including overlaps.
    novel_witnesses, novel_unknown = [], not complete or bool(potential)
    for later in evals:
        earlier = [c for c in cc if c['start'] < later['start'] and c['evaluation'] != 'excluded']
        completed = [c for c in earlier if c['evaluation'] == 'confirmed' and c['end'] < later['start']]
        if not completed:
            if earlier and set(later['groups']) - {g for c in earlier for g in c['groups']}:
                novel_unknown = True
            continue
        for g in later['groups']:
            previous = [c for c in earlier if g in c['groups']]
            if any(c['evaluation'] == 'confirmed' and c['end'] < later['start'] for c in previous): continue
            if previous or any(c['evaluation'] == 'unresolved' for c in earlier):
                novel_unknown = True
            else:
                novel_witnesses.append({'group': g, 'earlier_result': completed[0], 'later_command': later})
    result['motifs'][MOTIFS[0]] = indicator(any(len(c['groups']) >= 2 for c in evals),
        not complete or bool(potential), [c for c in evals if len(c['groups']) >= 2])
    result['motifs'][MOTIFS[1]] = indicator(bool(novel_witnesses), novel_unknown, novel_witnesses)
    decl = (evidence or {}).get('declarations')
    declarations_complete = ((decl.get('complete') is True if decl is not None else True) and messages_complete
                             and not any(m['id'] in unknown_items for m in r['messages']))
    declarations = decl.get('items', []) if decl is not None else [
        {'status': 'unresolved' if m['id'] in unknown_items else 'confirmed', 'kind': 'actual_revise' if 'primary_revise' in m['action'] else 'actual_adopt',
         'at': m['at'], 'ev': m['ev'], 'quote': m.get('quote'), 'raw_quote_ref': m.get('quote_ref'),
         'review_ref': 'audited-message-action:' + m['id'], 'item_id': m['id']}
        for m in r['messages'] if set(m['action']) & {'primary_adopt', 'primary_revise'}]
    for d in declarations:
        check(d.get('status') in ('confirmed', 'unresolved'), 'invalid declaration status')
        if d.get('item_id') not in unknown_items:
            _lines(d.get('ev'), nlines, 'declaration')
        check(d.get('kind') in ('actual_adopt', 'actual_revise'), 'planned selection is not a declaration')
        check(type(d.get('at')) is int and (d['at'] in d['ev'] or d.get('item_id') in unknown_items), 'actual declaration anchor required')
        check((d.get('quote') or d.get('item_id') in unknown_items) and d.get('review_ref'), 'declaration quote/review required')
    result['declarations'] = sorted(declarations, key=lambda d: (d['at'], d['kind']))
    confirmed = [d for d in declarations if d['status'] == 'confirmed']
    doubtful = [d for d in declarations if d['status'] != 'confirmed']
    no_decl = declarations_complete and not declarations
    result['declaration_status'] = ('no_declaration_observed' if no_decl else
                                    'confirmed' if declarations_complete and not doubtful else 'unresolved')
    after = [{'declaration': d, 'command': c} for d in confirmed for c in evals if d['at'] < c['start']]
    result['motifs'][MOTIFS[2]] = indicator(bool(after), not no_decl and (not declarations_complete or bool(doubtful) or
        bool(potential)), after, no_declaration=no_decl)
    last = max((d['at'] for d in confirmed), default=None)
    later_unknown = last is None or any(d['at'] >= last for d in doubtful) or not declarations_complete
    verification = [{'declaration': d, 'command': c} for d in confirmed if d['at'] == last
                    for c in cc if c['verification'] == 'confirmed' and last < c['start']]
    result['motifs'][MOTIFS[3]] = indicator(bool(verification) and not later_unknown,
        not no_decl and (later_unknown or any(c['verification'] == 'unresolved' for c in cc)),
        verification, no_declaration=no_decl)
    # Accepted independent declaration codes are supplied through an explicit adapter.
    choice_status = (reviewed_choice or {}).get('status', 'unresolved')
    check(choice_status in ('confirmed', 'unresolved', 'no_declaration_observed'), 'invalid final-choice status')
    choice_groups = (reviewed_choice or {}).get('groups', [])
    check(set(choice_groups) <= set(GROUPS), 'unknown reviewed final-choice group')
    if choice_status == 'confirmed':
        check(bool(choice_groups), 'confirmed final choice needs at least one group')
        if not external_choice:
            _lines(reviewed_choice.get('ev'), nlines, 'reviewed choice')
    result['reviewed_choice'] = reviewed_choice
    for g in GROUPS:
        result['declared'][g] = indicator(choice_status == 'confirmed' and g in choice_groups,
            choice_status == 'unresolved', [reviewed_choice] if choice_status == 'confirmed' and g in choice_groups else [],
            no_declaration=choice_status == 'no_declaration_observed')
    for criterion in CRITERIA:
        witnesses = [{'id': m['id'], 'at': m['at'], 'ev': m['ev'], 'quote': m['quote']}
                     for m in r['messages'] if criterion in m['criterion'] and m['id'] not in unknown_items]
        result['criteria'][criterion] = indicator(bool(witnesses) and not unresolved,
                                                  not messages_complete or unresolved, witnesses)
    confirmed_families = {d: sorted({f.split(':', 1)[1] for c in evals for f in c['families']
                                   if f.startswith(d + ':')}) for d in ('O', 'S', 'M', 'W', 'T', 'I')}
    result['family_portfolio'] = {'confirmed': confirmed_families, 'audited_portfolio': r['portfolio'],
                                  'unresolved_commands': [c['id'] for c in potential]}
    result['breadth'] = {'min': sum(len(confirmed_families[d]) >= 2 for d in ('O', 'S', 'M', 'W', 'T')),
                         'max': 5 if potential or not complete else sum(len(confirmed_families[d]) >= 2 for d in ('O', 'S', 'M', 'W', 'T'))}
    for name in scoped_features:
        field, key = name.split('.', 1)
        result[field][key]['state'] = 'unresolved'
        result[field][key]['reason'] = 'Root reporting-scope clearance identifies affected feature'
    return result


def summarize(cases):
    """No pooling across external strata; caller passes exactly one released facet."""
    check(not any(c['is_original'] for c in cases), 'original must be displayed separately')
    check(len({c['id'] for c in cases}) == len(cases), 'duplicate cases')
    n = len(cases)
    out = {'N': n, 'ids': sorted(c['id'] for c in cases), 'features': {}}
    for field, names in (('evaluated', GROUPS), ('declared', GROUPS), ('motifs', MOTIFS), ('criteria', CRITERIA)):
        out['features'][field] = {}
        for name in names:
            ids = {state: sorted(c['id'] for c in cases if c[field][name]['state'] == state) for state in STATES}
            p, u = len(ids['positive']), len(ids['unresolved'])
            out['features'][field][name] = {'P': p, 'U': u, 'D': len(ids['no_declaration_observed']),
                'negative': len(ids['negative']), 'N': n, 'lower': p / n if n else None,
                'upper': (p + u) / n if n else None, 'ids': ids}
    out['zero_command_ids'] = sorted(c['id'] for c in cases if not c['activity'])
    out['breadth'] = {f'{lo}:{hi}': sorted(c['id'] for c in cases if c['breadth'] == {'min': lo, 'max': hi})
                     for lo in range(6) for hi in range(lo, 6)}
    return out


def released_join(cases, strata, release):
    """Explicit Root gate. The immutable neutral features must already exist."""
    check(release.get('approved_by') == 'Root' and release.get('released') is True, 'Root release required')
    check(release.get('definitions_sha256') == DEFINITIONS_SHA256, 'release definitions mismatch')
    check(release.get('features_sha256') == cohort_digest(cases), 'release must bind exact neutral features')
    check(release.get('source_checks_complete') is True and release.get('review_ref'),
          'representative stronger source checks and release provenance required')
    replay_ids = {c['id'] for c in cases if not c['is_original']}
    check(set(strata) == replay_ids, 'stratum mapping must cover every replay exactly once')
    check(set(strata.values()) <= {'tail', 'body'}, 'only frozen tail/body strata allowed')
    expected = release.get('expected_counts', {'tail': 1142, 'body': 1858})
    check({s: sum(v == s for v in strata.values()) for s in ('tail', 'body')} == expected, 'allocation mismatch')
    return {s: summarize([c for c in cases if not c['is_original'] and strata[c['id']] == s])
            for s in ('tail', 'body')}


def select_examples(cases):
    return {motif: min((c['id'] for c in cases if not c['is_original'] and
                       c['motifs'][motif]['state'] == 'positive'),
                      key=lambda identifier: hashlib.sha256(('display-example-v1|' + identifier).encode()).hexdigest(),
                      default=None) for motif in MOTIFS}


def export_witnesses(cases, destination):
    """Exclusive-create outputs preserve previous figures/data and every case status."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    check(len({c['id'] for c in cases}) == len(cases), 'duplicate cases')
    ordered = sorted(cases, key=lambda c: c['id'])
    with (dest / 'all-cases.json').open('x') as f:
        json.dump({'definitions': DEFINITIONS, 'definitions_sha256': DEFINITIONS_SHA256,
                   'cases': ordered, 'examples': select_examples(ordered)}, f, sort_keys=True, indent=2)
        f.write('\n')
    with (dest / 'all-case-witnesses.csv').open('x', newline='') as f:
        fields = ['id', 'is_original', 'source_sha256', 'packet_sha256', 'annotation_sha256',
                  'feature_set', 'feature', 'state', 'reason', 'witnesses_json']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in ordered:
            for field in ('evaluated', 'declared', 'motifs', 'criteria'):
                for name, value in case[field].items():
                    writer.writerow({**{k: case[k] for k in fields[:5]}, 'feature_set': field, 'feature': name,
                        'state': value['state'], 'reason': value['reason'],
                        'witnesses_json': canonical(value['witnesses']).decode()})
    with (dest / 'all-case-command-activity.csv').open('x', newline='') as f:
        fields = ['id', 'source_sha256', 'command_id', 'rank', 'total_commands', 'position',
                  'case_weight', 'state', 'raw_main_purpose', 'annotation_uncertain', 'ev_json']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in ordered:
            for command in case['activity']:
                writer.writerow({'id': case['id'], 'source_sha256': case['source_sha256'],
                    'command_id': command['id'], **{k: command[k] for k in fields[3:-1]},
                    'ev_json': canonical(command['ev']).decode()})
    return {'features_sha256': cohort_digest(cases), 'cases': len(cases), 'definitions_sha256': DEFINITIONS_SHA256}
