#!/usr/bin/env python3
"""Build GPT-6-Luna's Card--Krueger records (module 08) from the research repository's run folders.

Run once by the authors; run_all.sh does not call it. No model calls; about 90 seconds.

Writes, under data/agent_runs/card_luna/:
  packet/                    data.csv, CODEBOOK.md, DESIGN.md and the three prompts every session received,
                             each checked against the hash the frozen initial state recorded
  original.json, original/, programs/original-analysis.R   the original run on all 410 stores
  replays.csv.gz             one row per rostered replay: subset, stratum, commitment, own verdict and
                             the reviewing agent's fact label
  replay-programs.jsonl.gz   each replay's final analysis.R, rebuilt from its recorded tool calls
  replay-results.csv         that program rerun on the replay's subset (data_generation/replay_reruns/)
  choice-codes.csv           coded employment measure, sample rule and specification of each declaration
  soundness-review.csv.gz    GPT-6.1-Sol's methodological review and Claude Opus 5.5's audit of it
  activity-steps.csv.gz      each replay's commands (with the annotators' purpose code) and file edits
  analysis-streams.jsonl.xz  the analysis-phase event stream of every session
  replay-requests/           tool calls and user messages of each replay's final model request
  (both with the cluster's host paths redacted by redact.py)
  review_prompt.json         what every fact-review pair carried besides the two declarations
  prompts/                   operator instruction and the review and audit prompts
and data/intermediate/08_card/card-bank-subsets.csv.gz and card-bank-p.csv (the reference bank).

Sources, in the research repository's output/ folder (the only argument):
  card-connect-repaired-original-execution-20261016/   the original session (initial state that every replay
                                                      cloned, events, model requests)
  final-simulation-runs-2026-10-15/core-preparation/accepted-runtime/studies/codex_cli/   commitment and
                                                      comparison prompts of the runtime
  card3000-bank-execution-20261017/ROSTER.json        replay subsets, strata and run order
  card3000-bulk-execution-20261017/seal-*/review/public/   the replays' committed declarations
  card-trace-coding-20261018-v1/                      own verdicts (extraction/PRIVATE_join.jsonl, read from
                                                      the replay records) and the final choice codes
  card3000-independent-review-20261017/final3000-v1/  first declaration review (classification status only)
  card-terra-subscription-pilot-20261029/rereview/    the fact-label review (arm luna-fact) and its pair map
  card-mn-robustness-20261027/                        bank files built by its code/01_prepare.R from the
                                                      bank archive; original analysis on the full data
  card-replay-estimates-20261029/                     model-request extracts, rerun results and checks
  card-soundness-review-20261029/, card-opus-audit-20261030/   review and audit records and prompts
  manuscript-rebuild-data/data/                       event streams and step annotations (card-semantic/)
Usage: python3 extract_records.py <research>/output
"""
import csv
import glob
import gzip
import hashlib
import io
import json
import shutil
import sys
import tomllib
from pathlib import Path

import rebuild_programs
import redact

SRC = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data' / 'agent_runs' / 'card_luna'
INTER = ROOT / 'data' / 'intermediate' / '08_card'
CASE = SRC / 'card-connect-repaired-original-execution-20261016' / 'sealed-export' / 'archive' / 'case'
MN = SRC / 'card-mn-robustness-20261027'
EST = SRC / 'card-replay-estimates-20261029'
REREVIEW = SRC / 'card-terra-subscription-pilot-20261029' / 'rereview'
CODING = SRC / 'card-trace-coding-20261018-v1'
REBUILD = SRC / 'manuscript-rebuild-data' / 'data'
DECL = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
COHORT = {'cohort': 'card_luna', 'model': 'gpt-6-luna', 'effort': 'high', 'route': 'api'}
FIELDS = (['session_id', 'cohort', 'role', 'model', 'effort', 'route', 'position', 'bank_id', 'bank_p', 'stratum',
           'status'] + [f'commit_{k}' for k in DECL] +
          ['own_match', 'own_reason', 'review_label', 'review_reason', 'review_session', 'first_review_status'])


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, fields, rows, compress=False):
    buffer = io.StringIO()
    w = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    if compress:
        with gzip.GzipFile(path, 'wb', mtime=0) as f:
            f.write(buffer.getvalue().encode())
    else:
        path.write_text(buffer.getvalue(), encoding='utf-8')


def packet():
    """The files every session received, from the original's frozen initial state (which each replay cloned with
    its own data.csv) and the runtime's prompt files, checked against the hashes that state records."""
    folder = OUT / 'packet'
    folder.mkdir(exist_ok=True)
    manifest = json.load(open(CASE / 'payload/snapshot-v1/manifest.json', encoding='utf-8'))['operator']
    recorded = json.load(open(CASE / 'payload/results/original/analysis/phase.json', encoding='utf-8'))['workspace_sha256']
    for name in ('data.csv', 'CODEBOOK.md', 'DESIGN.md'):
        shutil.copyfile(CASE / 'payload/snapshot-v1/tree/workspace' / name, folder / name)
        assert sha((folder / name).read_text(encoding='utf-8')) == recorded[name], name
    assert sha(manifest['initial_prompt']) == manifest['initial_prompt_sha256']
    (folder / 'research-prompt.txt').write_text(manifest['initial_prompt'], encoding='utf-8')
    runtime = SRC / 'final-simulation-runs-2026-10-15/core-preparation/accepted-runtime/studies/codex_cli'
    for name in ('commitment_prompt', 'comparison_prompt'):
        shutil.copyfile(runtime / f'{name}.txt', folder / f'{name}.txt')
        assert sha((folder / f'{name}.txt').read_text(encoding='utf-8')) == manifest[f'{name}_sha256'], name


def original():
    """The original session: declaration, final message, emitted program and the report files it wrote."""
    result = json.load(open(CASE / 'payload/results/original/result.json', encoding='utf-8'))
    phase = json.load(open(CASE / 'payload/results/original/analysis/phase.json', encoding='utf-8'))
    events = jsonl(CASE / 'payload/results/original/analysis/events.jsonl')
    final = [e['item']['text'] for e in events
             if e.get('type') == 'item.completed' and e['item'].get('type') == 'agent_message'][-1]
    fit = json.load(open(MN / 'results/original_analysis.json', encoding='utf-8'))
    one = lambda x: x[0] if isinstance(x, list) else x
    info = {'session_id': result['session_id'], **COHORT, 'role': 'original',
            'p_value': one(fit['p_value']), 'estimate': one(fit['estimate']), 'se': one(fit['se']),
            'commitment': {k: result['commitment'][k] for k in DECL}, 'final_report': final}
    info = {k: info[k] for k in ['session_id', 'cohort', 'role', 'model', 'effort', 'route', 'p_value', 'estimate',
                                 'se', 'commitment', 'final_report']}
    (OUT / 'original.json').write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    # The workspace files it wrote, replayed from the tool calls of its final model request; each must
    # equal the hash the run recorded for the file.
    requests = [json.load(open(p, encoding='utf-8')) for p in glob.glob(str(CASE / 'relay/original/provider-receipts/*/request.json'))]
    files = rebuild_programs.workspace(rebuild_programs.items(max(requests, key=lambda r: len(r['input']))))
    recorded = phase['workspace_sha256']
    for name, target in (('analysis.R', OUT / 'programs/original-analysis.R'),
                         ('analysis_report.md', OUT / 'original/analysis_report.md'),
                         ('explore.R', OUT / 'original/explore.R')):
        text = files['/workspace/' + name] + '\n'
        assert sha(text) == recorded[name], name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    assert sha((OUT / 'programs/original-analysis.R').read_text(encoding='utf-8')) == result['program']['sha256']
    return result['commitment']


def replays(target):
    roster = json.load(open(SRC / 'card3000-bank-execution-20261017/ROSTER.json', encoding='utf-8'))['selected_fresh_jobs']
    bank_p = {r['bank_id']: r['p_value'] for r in csv.DictReader(open(MN / 'processed/card-bank-p.csv', encoding='utf-8'))}
    join = {r['case_id']: r for r in jsonl(CODING / 'extraction/PRIVATE_join.jsonl')}
    pairs = {}
    for path in sorted(glob.glob(str(SRC / 'card3000-bulk-execution-20261017/seal-*/review/public/batch-*.json'))):
        for p in json.load(open(path, encoding='utf-8'))['pairs']:
            assert {k: p['target_declaration'][k] for k in DECL} == {k: target[k] for k in DECL}
            pairs.setdefault(p['blinded_id'], p['candidate_declaration'])
    own = {r['job_id']: r for r in json.load(open(SRC / 'card3000-independent-review-20261017/final3000-v1/raw-labels.json', encoding='utf-8'))['rows']}
    first = {r['job_id']: r['reviewed_status'] for r in
             json.load(open(SRC / 'card3000-independent-review-20261017/final3000-v1/reviewed-labels.json', encoding='utf-8'))['rows']}
    # Fact-label review: arm luna-fact, one completed session per batch of 60 blinded pairs.
    case_of = json.load(open(REREVIEW / 'luna-map.json', encoding='utf-8'))
    fact = {}
    for path in sorted(glob.glob(str(REREVIEW / 'out/results-*.jsonl'))):
        for r in jsonl(path):
            if r['arm'] == 'luna-fact' and r['status'] == 'complete':
                for x in r['labels']:
                    assert case_of[x['blinded_id']] not in fact
                    fact[case_of[x['blinded_id']]] = (x['label'], x['reason'], r['session_id'])
    joined = {r['case_id']: r for r in csv.DictReader(open(REREVIEW.parent / 'processed/rereview-luna-fact.csv', encoding='utf-8'))}
    rows = []
    for job in sorted(roster, key=lambda j: j['order_position']):
        cid, j = job['job_id'], join[job['job_id']]
        declaration = pairs[j['raw_identifiers']['blinded_id']]
        verdict = j['raw_self_verdict']
        assert j['raw_status'] == 'complete' and verdict['model_null_match'] == own[cid]['raw_match']
        label, reason, session = fact[cid]
        assert (label, reason) == (joined[cid]['label'], joined[cid]['reason'])
        rows.append({'session_id': cid, **COHORT, 'role': 'replay', 'position': job['order_position'],
                     'bank_id': job['world_id'], 'bank_p': bank_p[str(job['world_id'])], 'stratum': job['stratum'],
                     'status': 'completed', **{f'commit_{k}': declaration[k] for k in DECL},
                     'own_match': str(verdict['model_null_match']), 'own_reason': verdict['reason'],
                     'review_label': label, 'review_reason': reason, 'review_session': session,
                     'first_review_status': first[cid]})
    assert len(rows) == 3000
    write_csv(OUT / 'replays.csv.gz', FIELDS, rows, compress=True)
    batch = json.load(open(REREVIEW / 'public/luna-fact/batch-001.json', encoding='utf-8'))
    keep = {k: batch[k] for k in ('instruction', 'data_facts', 'target_declaration', 'public_context')}
    (OUT / 'review_prompt.json').write_text(json.dumps(keep, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {r['session_id'] for r in rows}, {cid: j['coding_id'] for cid, j in join.items()}


def programs_and_results(ids):
    """Final analysis.R of each replay, rebuilt from the recorded tool calls, and its rerun on the subset."""
    (OUT / 'replay-requests').mkdir(exist_ok=True)
    for path in sorted((EST / 'raw').glob('shard-*.jsonl.gz')):
        redact.requests(path, OUT / 'replay-requests' / path.name)
    rebuilt = rebuild_programs.rebuild_all(sorted((OUT / 'replay-requests').glob('shard-*.jsonl.gz')))
    saved = {r['replay_id']: r for r in jsonl(EST / 'processed/rebuilt-analysis-r.jsonl')}
    assert {k: (v['status'], v['analysis_R']) for k, v in saved.items()} == rebuilt and set(rebuilt) == ids
    with gzip.GzipFile(OUT / 'replay-programs.jsonl.gz', 'wb', mtime=0) as f:
        for k in sorted(rebuilt):
            status, text = rebuilt[k]
            f.write((json.dumps({'session_id': k, 'status': status, 'analysis_R': text}, ensure_ascii=False) + '\n').encode())
    runs = {r['replay_id']: r for r in csv.DictReader(open(EST / 'processed/replay-results.csv', encoding='utf-8'))}
    checks = {r['replay_id']: r for r in csv.DictReader(open(EST / 'processed/replay-validation.csv', encoding='utf-8'))}
    fields = ['session_id', 'estimate', 'se', 'p_value', 'error', 'estimate_printed', 'p_value_printed']
    rows = [{'session_id': k, **{f: runs[k][f] for f in ('estimate', 'se', 'p_value', 'error')},
             'estimate_printed': checks[k]['estimate_confirmed'] if k in checks else '',
             'p_value_printed': checks[k]['p_value_confirmed'] if k in checks else ''} for k in sorted(runs)]
    write_csv(OUT / 'replay-results.csv', fields, rows)


def choice_codes(coding_of_case):
    codes = {r['coding_id']: r['display'] for r in jsonl(CODING / 'integration/FINAL_CODES.jsonl')}
    rows = [{'session_id': case, 'measure': codes[coding]['employment_measure'],
             'sample_rule': codes[coding]['sample_treatment'], 'specification': codes[coding]['mean_model']}
            for case, coding in sorted(coding_of_case.items())]
    saved = {r['replay_id']: r for r in csv.DictReader(open(MN / 'processed/card-replay-choices.csv', encoding='utf-8'))}
    assert all({k: saved[r['session_id']][k] for k in ('measure', 'sample_rule', 'specification')} ==
               {k: r[k] for k in ('measure', 'sample_rule', 'specification')} for r in rows) and len(rows) == 3000
    write_csv(OUT / 'choice-codes.csv', ['session_id', 'measure', 'sample_rule', 'specification'], rows)


def soundness():
    review = {r['replay_id']: r for r in jsonl(SRC / 'card-soundness-review-20261029/runs/main/results.jsonl')}
    audit = {r['replay_id']: r for r in jsonl(SRC / 'card-opus-audit-20261030/runs/main/results.jsonl')}
    assert all(r['status'] == 'complete' for r in review.values()) and set(audit) <= set(review)
    assert all(r['status'] == 'complete' for r in audit.values())
    fields = ['session_id', 'review_model', 'review_pass', 'review_reason', 'audit_model', 'audit_major_error',
              'audit_reviewer_correct', 'audit_reason']
    rows = []
    for k in sorted(review):
        a = audit.get(k)
        rows.append({'session_id': k, 'review_model': review[k]['model'],
                     'review_pass': str(review[k]['verdict']['pass']).lower(), 'review_reason': review[k]['verdict']['reason'],
                     'audit_model': a['model'] if a else '',
                     'audit_major_error': str(a['verdict']['major_error']).lower() if a else '',
                     'audit_reviewer_correct': str(a['verdict']['reviewer_correct']).lower() if a else '',
                     'audit_reason': a['verdict']['reason'] if a else ''})
    write_csv(OUT / 'soundness-review.csv.gz', fields, rows, compress=True)


def activity_steps(coding_of_case):
    """Commands with their annotated main purpose and file edits with the paths they change, in event order."""
    with gzip.open(REBUILD / 'card-semantic/cases.json.gz', 'rt', encoding='utf-8') as f:
        cases = {c['id']: c for c in json.load(f) if not c['is_original']}
    case_of = {v: k for k, v in coding_of_case.items()}
    edits = rebuild_programs.file_edits(OUT / 'analysis-streams.jsonl.xz')
    start = lambda x: x['at'][0] if isinstance(x['at'], list) else x['at']
    rows = []
    for coding in sorted(cases, key=lambda c: case_of[c]):
        case = cases[coding]
        if not case['activity']:  # no usable annotation of this replay's commands
            continue
        purpose = {a['id']: a['raw_main_purpose'] for a in case['activity']}
        step, seen = 0, set()
        for x in sorted(case['event_items'], key=start):
            if x['kind'] == 'commands' and x['id'] in purpose:
                kind, code, paths = 'command', purpose[x['id']], ''
                seen.add(x['id'])
            elif x['kind'] == 'files':
                kind, code, paths = 'file_edit', '', ';'.join(edits[case_of[coding]][x['id']])
            else:
                continue
            step += 1
            rows.append({'session_id': case_of[coding], 'step': step, 'kind': kind, 'purpose': code, 'paths': paths})
        assert seen == set(purpose)
    write_csv(OUT / 'activity-steps.csv.gz', ['session_id', 'step', 'kind', 'purpose', 'paths'], rows, compress=True)


def prompts():
    folder = OUT / 'prompts'
    folder.mkdir(exist_ok=True)
    config = tomllib.loads((CASE / 'payload/snapshot-v1/tree/codex-home/config.toml').read_text(encoding='utf-8'))
    (folder / 'developer-instructions.txt').write_text(config['developer_instructions'] + '\n', encoding='utf-8')
    shutil.copyfile(SRC / 'card-soundness-review-20261029/inputs/review_prompt.txt', folder / 'soundness-review-prompt.txt')
    shutil.copyfile(SRC / 'card-opus-audit-20261030/inputs/audit_prompt.txt', folder / 'soundness-audit-prompt.txt')


def main():
    for d in (OUT / 'programs', OUT / 'original', INTER):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MN / 'processed/card-bank-subsets.csv.gz', INTER / 'card-bank-subsets.csv.gz')
    shutil.copyfile(MN / 'processed/card-bank-p.csv', INTER / 'card-bank-p.csv')
    redact.streams(REBUILD / 'card-process/analysis-streams.jsonl.xz', OUT / 'analysis-streams.jsonl.xz')
    packet()
    ids, coding = replays(original())
    programs_and_results(ids)
    choice_codes(coding)
    soundness()
    activity_steps(coding)
    prompts()
    print('records written for', len(ids), 'replays')


if __name__ == '__main__':
    main()
