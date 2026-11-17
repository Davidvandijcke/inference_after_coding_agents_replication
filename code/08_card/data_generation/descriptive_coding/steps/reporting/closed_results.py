#!/usr/bin/env python3
"""Zero-API reporting bridge over explicitly closed, extracted review roots.

Run on a compute node with source archives in node TMP. This module never opens
an active database, calls a model, submits a job, or edits a ledger/manuscript.
The resulting final objects are reporting adapters, not native storage receipts.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile

from features import (canonical, check, cohort_digest, derive_case, digest,
                      export_witnesses, GROUPS, MOTIFS, DEFINITIONS_SHA256,
                      indicator, released_join, summarize)
from reviewed_choice_adapter import (build_adapter, file_sha, jsonl_records,
                                     roster_ids)

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
VERSION = 'card-closed-review-reporting-adapter-v1'
RECOVERIES = {}
RECOVERY_INPUTS = []


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('xb') as f:
        f.write(canonical(value) + b'\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def load_recovery_indexes(paths, proof_paths):
    """Consume verified sidecars; original runtime failures remain unchanged."""
    proofs={file_sha(p):str(p) for p in proof_paths}
    for path in paths:
        doc=load(path)
        check(doc.get('schema')=='card-provider-annotation-recovery-v1', 'unknown recovery sidecar schema')
        bound_proofs=doc.get('closed_proof_sha256s') or [doc['closed_proof_sha256']]
        check(bound_proofs and all(h in proofs for h in bound_proofs),
              'recovery must bind every actual existing closed Core proof')
        index_provenance={'path':str(path),'sha256':file_sha(path),
            'closed_proof_sha256s':bound_proofs,
            'closed_proof_paths':[proofs[h]for h in bound_proofs],
            'ambiguities':doc.get('ambiguities',[])}
        RECOVERY_INPUTS.append(index_provenance)
        ambiguous={(r['role'],r['id'])for r in doc.get('ambiguities',[])}
        for row in doc['recoveries']:
            key=row['role'],row['id']
            check(key not in ambiguous and row['role']in ('A','B','QA'), 'ambiguous or unrecognized recovered response')
            response=(path.parent/row['response_path']).resolve()
            check(response.is_relative_to(path.parent.resolve()), 'recovery response escapes sidecar directory')
            check(file_sha(response)==row['response_sha256'], 'recovered response bytes changed')
            value=load(response)
            check(value['id']==row['id'] and value.get('hash',value.get('source_sha256'))==row['source_sha256'],
                  'recovered response identity/source mismatch')
            check(row.get('bindings'), 'recovery request/settled-response bindings required')
            check(all(b.get('http_status')==200 and b.get('relay_status')=='settled'for b in row['bindings']),
                  'recovery must preserve settled HTTP200 response proof')
            previous=RECOVERIES.get(key)
            check(previous is None or digest(load(previous['response']))==digest(value),
                  'distinct recovered model outputs require explicit source disposition')
            RECOVERIES[key]={'response':response,'record':row,'index':index_provenance}


def recovery_provenance(role,cid,path):
    row=RECOVERIES.get((role,cid))
    if row and row['response'].resolve()==Path(path).resolve():
        return {'schema':'card-provider-annotation-recovery-v1','record':row['record'],'index':row['index'],
                'original_CLI_failure_preserved':True,'new_model_call':False}
    return None


def completed_directories(roots, role, cid):
    candidates = []
    native = {'A': 'native/full-cohort', 'B': 'native/full-source-audit-v2', 'QA': 'stronger-qa'}[role]
    for root in roots:
        root = Path(root)
        # Roots are closed snapshots, never a running continuation directory.
        directories = list((root / 'attempts' / role / cid).glob('attempt-*'))
        old = root / native / cid
        if old.exists():
            directories.append(old)
        for directory in directories:
            result_path = directory / 'RESULT.json'
            if not result_path.exists():
                continue
            result = load(result_path)
            field = {'A': 'A_complete', 'B': 'B_complete', 'QA': 'call_completed'}[role]
            if result.get(field) or result.get('call_completed'):
                response = directory / 'response.json'
                if not response.exists():
                    response = directory / ('A' if role == 'A' else 'B') / 'response.json'
                if response.exists():
                    candidates.append((response, directory))
    recovered=RECOVERIES.get((role,cid))
    # An accepted native/API response remains authoritative. A failed turn's
    # provider output must never become an alternate review of a completed case.
    if recovered and not candidates:
        candidates.append((recovered['response'],recovered['response'].parent))
    # JSON formatting alone does not create distinct model outputs. Every source
    # byte hash remains preserved; differing model fields stay ambiguous.
    unique = {}
    for response, directory in candidates:
        unique.setdefault(digest(load(response)), (response, directory))
    check(len(unique) <= 1, f'{cid} {role}: distinct completed responses need explicit source selection')
    return next(iter(unique.values()), None)


def unkey(value):
    # Structural conversion only; preserve every model-selected label/reference.
    value = json.loads(json.dumps(value))
    for parent in ('annotation', 'item_audit'):
        for group in ('commands', 'messages', 'files'):
            rows = value[parent][group]
            if isinstance(rows, dict):
                check(all(k == row['id'] for k, row in rows.items()), 'source item key/id mismatch')
                value[parent][group] = list(rows.values())
    return value


def scoped_expansion(raw, cid, store, refs):
    """Resolve each selected reference independently, retaining bad refs unchanged."""
    _, aliases = refs.compact_view(cid, store)
    source_map = aliases['source_map']
    refs._verify_map(source_map)
    check(aliases['alias_map_sha256'] == refs._alias_digest(aliases), 'alias-map hash mismatch')
    expanded = json.loads(json.dumps(raw))
    bad, lookups = {}, []
    def exact_lookup(row, group=None, link=False):
        check('quote' not in row, 'ambiguous quote and quote_ref')
        selected = row['quote_ref']
        check(selected in aliases['aliases'], 'unknown model-selected source alias: ' + str(selected))
        span = source_map['spans'][aliases['aliases'][selected]]
        check(span['citeable'], 'uncitable model-selected source span')
        if link:
            check(span['event_line'] in set(row['from'] + row['to']), 'selected link reference absent from model event sets')
        else:
            check(span['item_id'] == row['id'] and span['group'] == group, 'model-selected reference belongs to another item/group')
        result = dict(row); result['quote'] = span['text']; del result['quote_ref']
        lookups.append({'model_selected_ref':selected, 'full_ref':span['ref'], 'item_id':span['item_id'],
                        'event_line':span['event_line'], 'quote':span['text'], 'span_sha256':span['sha256'],
                        'source_map_sha256':source_map['map_sha256'], 'host_selected_evidence':False})
        return result
    for parent in ('annotation', 'item_audit'):
        for group in ('commands', 'messages', 'files'):
            for n, row in enumerate(raw[parent][group]):
                if 'quote_ref' not in row:
                    continue
                try:
                    expanded[parent][group][n] = exact_lookup(row, group)
                except (ValueError, KeyError, TypeError) as error:
                    bad.setdefault(row['id'], []).append({'path': f'{parent}.{group}[{n}]',
                        'model_quote_ref': row.get('quote_ref'), 'error': str(error)})
    for n, row in enumerate(raw['annotation'].get('links', [])):
        if 'quote_ref' not in row: continue
        try:
            expanded['annotation']['links'][n] = exact_lookup(row, link=True)
        except (ValueError, KeyError, TypeError) as error:
            bad.setdefault('source-link-' + str(n), []).append({'path': f'annotation.links[{n}]',
                'model_quote_ref': row.get('quote_ref'), 'error': str(error)})
    return expanded, bad, lookups


def item_source_validation(annotation, audit, prediction, store, api, expansion_errors):
    """Validate reported typed fields and own-item evidence, preserving all defects."""
    cid = annotation['id']
    packet, _ = store.get(cid)
    check(annotation['hash'] == packet['eventstream_sha256'] == audit['source_sha256'], 'source hash mismatch')
    check(prediction['id'] == cid and prediction['hash'] == annotation['hash'], 'raw A identity/source mismatch')
    check(audit['id'] == cid and audit['prediction_sha256'] == digest(prediction), 'A/B prediction binding mismatch')
    roster = store.item_roster(cid)
    schema = api.production_schema(store.vocab, store.extra_enums)
    bad = dict(expansion_errors)
    structural = []
    for group in ('commands','messages','files'):
        rows = annotation[group]
        expected = {i for i,r in roster.items() if api.GROUPS[r['type']] == group}
        check(len(rows) == len({r['id'] for r in rows}) and {r['id'] for r in rows} == expected,
              'full exact annotation item roster required')
        audited = audit['item_audit'][group]
        check(len(audited) == len({r['id'] for r in audited}) and {r['id'] for r in audited} == expected,
              'full exact audit item roster required')
        audits = {r['id']: r for r in audited}
        for row in rows:
            iid = row['id']
            # Reportable semantic enums are consumed exactly; malformed fields
            # cannot be repaired algorithmically and make the case unavailable.
            needed = {'commands':['purpose','main_purpose','exec','families','dim'],
                      'messages':['criterion','action','ref'], 'files':[]}[group]
            for key in needed:
                api.schema_validate(row[key], schema['properties'][group]['items']['properties'][key])
            try:
                store.evidence(cid, row, check_quote=False)
                if group in ('commands','messages'): api.item_quote_check(annotation, row, store, group)
                else: store.evidence(cid, row)
                check(any(n in row['ev'] for n in roster[iid]['occurrence_lines']), 'own-item evidence required')
                ar = audits[iid]
                store.evidence(cid, ar, check_quote=False)
                api.item_quote_check(annotation, ar, store, group)
                check(ar.get('decision') in ('confirm','correct','unresolved') and ar.get('reason'), 'explicit audit disposition/reason required')
                check(any(n in ar['ev'] for n in roster[iid]['occurrence_lines']), 'audit own-item evidence required')
            except (ValueError, KeyError, TypeError) as error:
                bad.setdefault(iid, []).append({'error': str(error), 'path': group + ':' + iid})
            if audits[iid].get('decision') == 'unresolved':
                bad.setdefault(iid, []).append({'error':'explicit unresolved per-item audit disposition'})
    return {'valid': not bad, 'mode':'root_item_scoped_source_validation',
            'invalid_item_ids':sorted(i for i in bad if i in roster),
            'other_pending_ids':sorted(i for i in bad if i not in roster),
            'item_errors':bad, 'structural_errors':structural,
            'all_source_items_examined':True, 'semantic_correctness_established':False}


def scope_from_policy(final, source_hash, source_state, policy):
    annotation = final['annotation']
    commands = {r['id'] for r in annotation['commands']}
    messages = {r['id'] for r in annotation['messages']}
    files = {r['id'] for r in annotation['files']}
    decisions = []
    for claim in final['unresolved_adjudication']:
        iid = claim['id']
        features = []
        affected = [iid] if iid in commands | messages | files else []
        if not affected: features = ['motifs.' + x for x in MOTIFS[2:]]
        decisions.append({'id':iid, 'scope':'affected', 'reason':
            'Root measurement policy scopes raw pending claim by exact roster ID; no semantic adjudication.',
            'affected_item_ids':affected, 'affected_features':features})
    return {'id':annotation['id'], 'source_sha256':source_hash,
            'annotation_sha256':final['annotation_sha256'], 'final_sha256':digest(final),
            'approved_by':'Root', 'review_ref':'ROOT_SCOPE_POLICY.json',
            'policy_sha256':digest(policy), 'pending_claims':decisions}


def reporting_final(cid, roots, store, api, refs):
    """Validate existing model references; never choose replacement references."""
    a = completed_directories(roots, 'A', cid)
    b = completed_directories(roots, 'B', cid)
    state = {'id': cid, 'A_response': bool(a), 'B_response': bool(b), 'valid': False}
    if not a or not b:
        state['reason'] = 'missing completed A or full-source B response'
        return None, state, None
    apath, _ = a
    bpath, bdir = b
    state.update(A_sha256=file_sha(apath), B_sha256=file_sha(bpath),
                 A_path=str(apath), B_path=str(bpath))
    state['provider_recoveries']={role:r for role,path in [('A',apath),('B',bpath)]
        if (r:=recovery_provenance(role,cid,path)) is not None}
    try:
        prediction, raw = load(apath), load(bpath)
        raw = unkey(raw)
        expanded, expansion_errors, lookup = scoped_expansion(raw, cid, store, refs)
        # Unused relation fields do not control basic command/message metrics.
        # Their raw labels/quotes remain unchanged in the reporting annotation.
        canonical_input = json.loads(json.dumps(expanded['annotation']))
        raw_links = canonical_input.get('links', [])
        canonical_input['links'] = []
        annotation, reconciliation = api.canonicalize_record(canonical_input, store, deduplicate_sets=True)
        annotation['links'] = raw_links
        reconciliation['unused_raw_links_preserved_without_adjudication'] = True
        validation = item_source_validation(annotation, expanded, prediction, store, api, expansion_errors)
        # Preserve the existing pipeline's complete diagnostic list. Unsupported
        # links and other unused fields are diagnostics, not whole-case gates.
        validation['full_pipeline_diagnostics'] = api.collect_audit_errors(
            prediction, expanded, store, deduplicate_sets=True)
        pending_by_id = {r['id']:r for r in expanded['unresolved_adjudication']}
        for iid, errors in validation['item_errors'].items():
            pending_by_id.setdefault(iid, {'id':iid, 'status':'unresolved',
                'reason':'Root item-scoped source-validation defect; exact raw errors retained in validation ledger.'})
        pending = list(pending_by_id.values())
        final = {'schema': 'card-final-v3', 'annotation': annotation,
                 'annotation_sha256': digest(annotation), 'unresolved_adjudication': pending,
                 'status': 'unresolved' if pending else 'audited',
                 'artifacts': {'pass_A_sha256': digest(prediction),
                               'source_audit_sha256': digest(expanded)},
                 'reporting_adapter': {'schema': VERSION, 'raw_sources': state.copy(),
                                       'native_storage_receipt_claimed': False}}
        state.update(valid=validation['valid'], reporting_usable=True, pending_claims=pending, final_sha256=digest(final),
                     annotation_sha256=digest(annotation),
                     validation=validation, derived_reconciliation=reconciliation)
        return final, state, {'prediction': prediction, 'audit': expanded,
                             'reference_lookup': lookup}
    except (ValueError, KeyError, TypeError, IndexError) as error:
        state.update(reason='source/structure validation failed; raw labels preserved',
                     error_type=type(error).__name__, error=str(error))
        return None, state, None


def missing_choice(case, choice):
    """External declaration coding is independent of missing process annotation."""
    if choice is None:
        return case
    check(choice['id'] == case['id'] and choice['source_sha256'] == case['source_sha256'],
          'external choice identity/source mismatch')
    check(choice.get('source_kind') == 'external_reviewed_declaration', 'external choice required')
    for name in ('sha256', 'record_sha256'):
        check(len(choice['source_artifact'][name]) == 64, 'exact external declaration provenance required')
    case['reviewed_choice'] = choice
    case['declared'] = {g: indicator(choice['status'] == 'confirmed' and g in choice['groups'],
        choice['status'] == 'unresolved', [choice] if g in choice['groups'] else []) for g in GROUPS}
    return case


def compare_qa(cases, roots, qa_ids, source_hashes, store=None, refs=None):
    """Compare the fixed selection; disagreement is review evidence, never a vote."""
    check(len(qa_ids) == 96 and len(set(qa_ids)) == 96, 'fixed96 QA IDs required')
    by_id = {c['id']: c for c in cases}
    rows = []
    for cid in qa_ids:
        selected = completed_directories(roots, 'QA', cid)
        row = {'id': cid, 'completed': bool(selected), 'source_valid': False, 'differences': []}
        if selected:
            response, directory = selected
            qa = load(response)
            validation_path = directory / 'SOURCE_VALIDATION.json'
            validation = load(validation_path) if validation_path.exists() else None
            recovery=recovery_provenance('QA',cid,response)
            if recovery is not None:
                # Invoke only the frozen pure schema/source-check functions.
                # No runner main(), model_call(), subprocess or archived command.
                qa_api=module(BASE/'stronger-qa/run_qa.py','closed_recovered_QA_source_validation')
                packet,_=store.get(cid);roster=store.item_roster(cid)
                _,aliases=refs.compact_view(cid,store)
                validation,_=qa_api.validate_source(qa,qa_api.schema(cid,packet['eventstream_sha256'],roster),aliases,roster)
                row['provider_response_recovery']=recovery
            row.update(response_sha256=file_sha(response), response_path=str(response),
                       source_validation=validation)
            row['source_valid'] = (qa.get('source_sha256') == source_hashes[cid] and
                                   (validation or {}).get('mechanically_valid') is True)
            if row['source_valid']:
                for field, keys in (('evaluated', GROUPS), ('motifs', MOTIFS)):
                    for key in keys:
                        item = qa['evaluated_groups'][key] if field == 'evaluated' else qa['motifs'][key]['state']
                        if field == 'motifs' and qa.get('no_declaration_observed') and key in MOTIFS[2:]:
                            item = 'no_declaration_observed'
                        actual = by_id[cid][field][key]['state']
                        if actual != item:
                            row['differences'].append({'feature_set': field, 'feature': key,
                                'reporting_state': actual, 'stronger_QA_state': item,
                                'stronger_witness': qa['motifs'][key] if field == 'motifs' else item})
            row['raw_QA'] = qa
        rows.append(row)
    return {'schema': VERSION, 'fixed_selection_count': 96, 'selection_sha256': digest(qa_ids),
            'completed': sum(r['completed'] for r in rows),
            'source_valid': sum(r['source_valid'] for r in rows),
            'case_disagreements': sum(bool(r['differences']) for r in rows), 'cases': rows,
            'semantic_agreement_is_not_truth': True, 'new_QA_calls': 0}


def count_table(facets, path):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['stratum','feature_set','feature','P','U','D','negative','N','lower','upper'])
        w.writeheader()
        for stratum, summary in facets.items():
            for field, rows in summary['features'].items():
                for key, row in rows.items():
                    w.writerow({'stratum': stratum,'feature_set': field,'feature': key,
                                **{k: row[k] for k in w.fieldnames[3:]}})


def compact_bundle(destination, inputs, cases, qa, facets, receipt):
    """Portable derived inputs; full source stays in its SHA-bound GL archives."""
    entries = {'definitions.json': {'definitions_sha256': DEFINITIONS_SHA256},
               'closed-inputs.json': inputs, 'fixed96-comparison.json': qa,
               'facets.json': facets, 'integration-status.json':receipt,
               'reproduce.py': 'import json,gzip,sys\nfrom features import summarize,cohort_digest\n'
                   'd=json.load(gzip.open(sys.argv[1],"rt"));strata=json.load(open("strata.json"));print(cohort_digest(d))\n'
                   'print(json.dumps({s:summarize([c for c in d if not c["is_original"] '
                   'and strata[c["id"]]==s]) for s in ("tail","body")}))\n'}
    # Source witnesses, all feature states, raw typed labels and exact external
    # declaration codes are already retained in these case objects.
    entries['cases.json.gz'] = gzip.compress(canonical(cases), mtime=0)
    entries['strata.json'] = inputs['strata']
    for name in ('ROOT_SCOPE_POLICY.json',):
        entries[name] = (destination / name).read_bytes()
    for name in ('features.py', 'render.py', 'closed_results.py', 'reviewed_choice_adapter.py',
                 'manuscript.py', 'render_saved_results.py'):
        entries[name] = (HERE / name).read_bytes()
    path = destination / 'card-process-reproducibility.tar.gz'
    with path.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tf:
            for name, value in sorted(entries.items()):
                data = value if isinstance(value, bytes) else value.encode() if isinstance(value, str) else canonical(value)
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644; info.mtime = 0
                tf.addfile(info, io.BytesIO(data))
    return {'path': str(path), 'sha256': file_sha(path), 'bytes': path.stat().st_size,
            'under_20MB': path.stat().st_size < 20_000_000, 'raw_traces_included': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--closed-root', action='append', required=True, type=Path)
    p.add_argument('--packets', required=True, type=Path)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--final-codes', required=True, type=Path)
    p.add_argument('--case-map', required=True, type=Path)
    p.add_argument('--qa-ids', required=True, type=Path)
    p.add_argument('--scope-clearances', type=Path)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--release', type=Path)
    p.add_argument('--qa-clearance', type=Path)
    p.add_argument('--closed-inputs', required=True, type=Path,
                   help='Core receipt listing closed archives and verified hashes')
    p.add_argument('--recovery-index',action='append',default=[],type=Path)
    p.add_argument('--recovery-closed-proof',action='append',default=[],type=Path)
    p.add_argument('--destination', required=True, type=Path)
    args = p.parse_args()
    check(os.environ.get('SLURM_JOB_ID'), 'full source processing belongs on compute TMP')
    dest = args.destination
    check(not dest.exists(), 'use a fresh immutable output directory')
    dest.mkdir(parents=True)
    load_recovery_indexes(args.recovery_index,args.recovery_closed_proof)
    api = module(BASE / 'pipeline/card_review.py', 'closed_reporting_card_review')
    refs = module(BASE / 'pipeline/quote_refs.py', 'closed_reporting_quote_refs')
    store = api.PacketStore(args.packets, args.raw, production=True).preload_manifest(args.manifest)
    declarations, dprov = jsonl_records(args.final_codes)
    roster, rprov = roster_ids(args.case_map)
    sources, sprov = jsonl_records(args.packets / 'INDEX.jsonl')
    choices = build_adapter(declarations, roster, sources)
    choices_by_id = choices['replays_by_coding_id']
    scopes = load(args.scope_clearances) if args.scope_clearances else {}
    evidence = load(args.evidence) if args.evidence else {}
    policy = load(HERE / 'ROOT_SCOPE_POLICY.json')
    save(dest / 'ROOT_SCOPE_POLICY.json', policy)
    cases, states, finals = [], [], []
    raw_file = gzip.GzipFile(filename=str(dest / 'raw-completed-review-records.jsonl.gz'), mode='xb', mtime=0)
    for source, _ in sorted(sources, key=lambda r: r[0]['coding_id'] or r[0]['case_id']):
        cid = source['coding_id'] or source['case_id']
        packet, (_, lines, _) = store.get(cid)
        metadata = {**source, 'coding_id':cid, 'line_count': len(lines)}
        final, state, raw_bindings = reporting_final(cid, args.closed_root, store, api, refs)
        # Raw completed responses stay verbatim as parsed JSON; source hashes and
        # byte hashes bind their original archive paths. Invalid outputs remain.
        raw_record = {'id':cid, 'sources':state.copy()}
        for role in ('A','B'):
            selected = completed_directories(args.closed_root, role, cid)
            if selected:
                raw_record[role] = {'response':load(selected[0]), 'sha256':file_sha(selected[0]), 'path':str(selected[0])}
        raw_file.write(canonical(raw_record) + b'\n')
        states.append(state)
        if final:
            scopes.setdefault(cid, scope_from_policy(final, source['eventstream_sha256'], state, policy))
            metadata['source_validation'] = {'valid': state['validation']['valid'],
                'mode': 'root_item_scoped_source_validation', 'approved_by':'Root',
                'scope_policy_sha256':digest(policy),
                'invalid_item_ids':state['validation']['invalid_item_ids'],
                'annotation_sha256': digest(final['annotation']),
                'source_sha256': source['eventstream_sha256'], 'final_sha256': digest(final),
                'review_ref': 'VALIDATION_LEDGER.jsonl#' + cid}
            finals.append({'id': cid, 'final': final, 'packet': metadata,
                           'scope_clearance': scopes.get(cid), 'evidence': evidence.get(cid)})
        choice = choices_by_id.get(cid)
        case = derive_case(final, metadata, choice, evidence.get(cid), scopes.get(cid))
        case['provider_annotation_recoveries']=state.get('provider_recoveries',{})
        if final is None:
            case = missing_choice(case, choice)
        cases.append(case)
    raw_file.close()
    check(len(cases) == 3001 and sum(not c['is_original'] for c in cases) == 3000,
          'all3000 plus separate original required')
    with (dest / 'VALIDATION_LEDGER.jsonl').open('xb') as f:
        for row in states: f.write(canonical(row) + b'\n')
    with gzip.GzipFile(filename=str(dest / 'validated-reporting-inputs.jsonl.gz'), mode='xb', mtime=0) as f:
        for row in finals: f.write(canonical(row) + b'\n')
    qa = compare_qa(cases, args.closed_root, load(args.qa_ids),
                    {r['coding_id'] or r['case_id']: r['eventstream_sha256'] for r, _ in sources},store,refs)
    save(dest / 'FIXED96_COMPARISON.json', qa)
    feature_receipt = export_witnesses(cases, dest)
    with args.case_map.open(newline='') as f:
        strata = {r['coding_id']: r['stratum'] for r in csv.DictReader(f)}
    # Root authorized this frozen join. Counts remain explicitly provisional
    # until the exact-feature and stronger-source release passes below.
    check({s: sum(v == s for v in strata.values()) for s in ('tail','body')} == {'tail':1142,'body':1858}, 'allocation mismatch')
    facets = {s:summarize([c for c in cases if not c['is_original'] and strata[c['id']] == s]) for s in ('tail','body')}
    save(dest / 'all3000-counts.json', facets)
    count_table(facets, dest / 'all3000-counts.csv')
    inputs = {'schema': VERSION, 'closed_archives': load(args.closed_inputs),
              'source_epoch': store.epoch, 'declarations': dprov, 'roster': rprov,
              'source_index': sprov, 'strata': strata, 'scope_clearances': scopes,
              'reporting_evidence': evidence, 'features_sha256': cohort_digest(cases),
              'provider_recovery_inputs':RECOVERY_INPUTS,
              'closed_review_artifacts': {n:{'sha256':file_sha(dest/n), 'bytes':(dest/n).stat().st_size}
                  for n in ('raw-completed-review-records.jsonl.gz','validated-reporting-inputs.jsonl.gz','VALIDATION_LEDGER.jsonl')}}
    save(dest / 'REPORTING_INPUTS.json', inputs)
    receipt = {**feature_receipt, 'schema': VERSION, 'replays':3000, 'original_separate':1,
        'validated_B':sum(r['valid'] for r in states), 'completed_B':sum(r['B_response'] for r in states),
        'item_scoped_reporting_B':sum(bool(r.get('reporting_usable')) for r in states),
        'provider_recovered_B':sum('B'in r.get('provider_recoveries',{})for r in states),
        'provider_recovered_QA':sum('provider_response_recovery'in r for r in qa['cases']),
        'fixed96_completed':qa['completed'], 'fixed96_source_valid':qa['source_valid'],
        'fixed96_disagreement_cases':qa['case_disagreements'], 'figures_released':False,
        'partial_cohort_review':not all(r['B_response'] for r in states),
        'model_calls':0, 'archived_commands_executed':0, 'ledger_writes':0}
    if args.release:
        check(all(r['B_response'] for r in states), 'all3001 B responses required before final reporting')
        check(qa['completed'] == 96, 'fixed96 stronger QA must close before final reporting')
        clearance = load(args.qa_clearance) if args.qa_clearance else {}
        check(clearance.get('approved_by') == 'Root' and clearance.get('comparison_sha256') == digest(qa)
              and clearance.get('review_ref'), 'Root source comparison disposition required')
        facets = released_join(cases, strata, load(args.release))
        from render import render_main, render_appendix
        render_main(facets, dest)
        activity = render_appendix(facets, {s:[c for c in cases if not c['is_original'] and strata[c['id']]==s]
                                          for s in ('tail','body')}, dest)
        save(dest / 'appendix-activity.json', activity)
        receipt['figures_released'] = True
    receipt['portable_bundle'] = compact_bundle(dest, inputs, cases, qa, facets, dict(receipt))
    save(dest / 'INTEGRATION_RECEIPT.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k != 'portable_bundle'}, sort_keys=True))


if __name__ == '__main__':
    main()
