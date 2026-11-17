#!/usr/bin/env python3
"""Build the compact agent records of the four other-agent Card--Krueger cohorts (Table tab:card_terra).

Writes data/agent_runs/card_terra/, card_sol/, card_sonnet/ and card_sonnet_masked/: one row per rostered
replay (replays.csv.gz), the reviewer's repeated reads (review_repeats.csv), the original's declaration and
report (original.json, original/), what the reviewer was given besides the pairs (review_prompt.json), its emitted program where it is kept locally (programs/), and the
keyword-rule flags and counts behind the table note's decline and naming rates. Also copies the GPT
originals' reference-bank p-values to data/intermediate/09_card_agents/.

Sources, in the research repository's output/ folder (given as the only argument):
  card-terra-subscription-pilot-20261029/   roster, runner records (runs/main/results.jsonl and
                                            runs/main/gl/out/results-*.jsonl), joined review labels
                                            (processed/rereview-terra-fact.csv), bank/terra-bank-p.csv, and the
                                            run's environment audit (processed/terra-replays.csv: machine,
                                            read isolation, refused and outside commands;
                                            processed/tmp-exposure.json: replays that listed or read another
                                            replay's files in a shared /tmp), shipped as five more columns
                                            of card_terra/replays.csv.gz
  card-sol-subscription-pilot-20261029/     roster, runner records (processed/replays*.jsonl, copies of
                                            the cluster records), processed/sol-fact*.csv, bank/sol-bank-p.csv
  card-sonnet-subscription-20261103/        processed/replays.jsonl (compact runner records),
                                            processed/sonnet-fact*.csv, processed/original-result.json
  card-sonnet-masked-20261104/pilot/        node-results/*.results.jsonl (runner records),
                                            processed/sonnet-masked-fact-3000*.csv, original/result.json
  card-recognition-check-20261104/results/  keyword-scan outputs (decline flags per Sonnet session;
                                            naming and Sol counts as aggregates)
  multi-model-originals-20261030/results/sol_card_facts.csv   blind reads of 831 Sol replay reports
The review labels come from the join scripts archived beside this file, which map the reviewer's blinded
pair ids back to cases. No model calls. Usage: python3 extract_records.py <research>/output
"""
import csv
import gzip
import io
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

SRC = Path(sys.argv[1])
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data' / 'agent_runs'
BANKS = ROOT / 'data' / 'intermediate' / '09_card_agents'
TERRA = SRC / 'card-terra-subscription-pilot-20261029'
SOL = SRC / 'card-sol-subscription-pilot-20261029'
SONNET = SRC / 'card-sonnet-subscription-20261103'
MASKED = SRC / 'card-sonnet-masked-20261104' / 'pilot'
RECOG = SRC / 'card-recognition-check-20261104' / 'results'
DECL = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
FIELDS = ['session_id', 'cohort', 'role', 'model', 'effort', 'route', 'account', 'position', 'bank_id', 'bank_p',
          'stratum', 'status'] + [f'commit_{k}' for k in DECL] + ['own_match', 'own_reason', 'review_label', 'review_reason']
# The Terra cohort's environment audit (its two defects are described in the module README and its codebook)
TERRA_FIELDS = FIELDS + ['machine', 'read_isolation', 'sandbox_refused', 'outside_commands', 'tmp_exposed']
MACHINE = {'mac': 'laptop', 'great_lakes': 'cluster'}
PLANNED = {'card_terra': 'gpt-6.1-terra', 'card_sol': 'gpt-6.1-sol', 'card_sonnet': 'claude-sonnet-5-5',
           'card_sonnet_masked': 'claude-sonnet-5-5'}
WORKSPACE = re.compile(r'(/[^\s/()\[\]]+)+?/ws-[0-9a-f]+/')


def scrub(text):
    """Agents' sandbox paths keep only the part inside the session's own workspace."""
    return WORKSPACE.sub('', text) if isinstance(text, str) else text


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]


def route_account(route):
    route, _, account = route.partition(':')
    return route, account or 'primary'


def by_session(path):
    """Rows of a joined CSV with its case id named session_id, as in replays.csv.gz."""
    return [{('session_id' if k == 'case_id' else k): v for k, v in r.items()} for r in csv.DictReader(open(path, newline='', encoding='utf-8'))]


def labels(path):
    return {r['case_id']: r for r in csv.DictReader(open(path, newline='', encoding='utf-8'))}


def row(cohort, case, rec, review):
    route, account = route_account(rec.get('route', '')) if rec else ('', '')
    complete = bool(rec) and rec['status'] == 'complete'
    verdict = (rec or {}).get('verdict') or {}
    commitment = (rec or {}).get('commitment') or {}
    lab = review.get(case['case_id'], {})
    models = (rec or {}).get('session_models') or [PLANNED[cohort]]
    efforts = (rec or {}).get('session_efforts') or ([rec['effort']] if rec and rec.get('effort') else ['high'])
    return {'session_id': case['case_id'], 'cohort': cohort, 'role': 'replay', 'model': '|'.join(models),
            'effort': '|'.join(efforts), 'route': route, 'account': account, 'position': case['position'],
            'bank_id': case['bank_id'], 'bank_p': repr(float(case['bank_p'])), 'stratum': case['stratum'],
            'status': 'completed' if complete else 'unavailable',
            **{f'commit_{k}': scrub(commitment.get(k, '')) if complete else '' for k in DECL},
            'own_match': str(bool(verdict['model_null_match'])) if complete else '',
            'own_reason': scrub(verdict.get('reason', '')) if complete else '',
            'review_label': lab.get('label', ''), 'review_reason': lab.get('reason', '')}


def write_csv_gz(path, rows, fields=FIELDS):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    with open(path, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as f:
        f.write(buf.getvalue().encode())


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def last_message(events_path):
    """The original's final report: the last agent message of its analysis turn (Codex event stream)."""
    msgs = [e['item']['text'] for e in jsonl(events_path)
            if e.get('type') == 'item.completed' and (e.get('item') or {}).get('type') == 'agent_message']
    return msgs[-1]


def write_original(folder, info):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'original.json').write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def review_prompt(batch_dir, out):
    """What every reviewer pair carried besides the two declarations: instruction, data facts, public context."""
    b = json.load(open(sorted(batch_dir.glob('batch-*.json'))[0], encoding='utf-8'))
    keep = {k: b[k] for k in ('instruction', 'data_facts', 'target_declaration', 'public_context')}
    (out / 'review_prompt.json').write_text(json.dumps(keep, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def gpt_original(run, cohort, out, files):
    """GPT originals keep their workspace in runs/original/session.tar.gz; ship its report files scrubbed."""
    result = json.load(open(run / 'runs/original/result.json', encoding='utf-8'))
    bank = json.load(open(run / 'bank/summary.json', encoding='utf-8'))
    write_original(out, {'session_id': result['session_id'], 'cohort': cohort, 'role': 'original',
                         'model': result['model'], 'effort': result['reasoning_effort'], 'route': 'chatgpt_subscription',
                         'p_value': bank['u'], 'estimate': bank['full_data_estimate'],
                         'commitment': result['commitment'], 'final_report': scrub(last_message(run / 'runs/original/analysis/events.jsonl'))})
    (out / 'programs').mkdir(exist_ok=True)
    shutil.copyfile(run / 'runs/original/emitted-analysis.R', out / 'programs' / 'original-analysis.R')
    (out / 'original').mkdir(exist_ok=True)
    with tarfile.open(run / 'runs/original/session.tar.gz') as t:
        for name in files:
            text = t.extractfile(f'workspace/{name}').read().decode()
            (out / 'original' / Path(name).name).write_text(scrub(text), encoding='utf-8')
    shutil.copyfile(run / 'bank' / f'{cohort.split("_")[1]}-bank-p.csv', BANKS / f'{cohort.split("_")[1]}_bank_p.csv')


def terra():
    out = OUT / 'card_terra'
    out.mkdir(parents=True, exist_ok=True)
    roster = json.load(open(TERRA / 'main-roster.json', encoding='utf-8'))
    records = jsonl(TERRA / 'runs/main/results.jsonl') + [r for p in sorted((TERRA / 'runs/main/gl/out').glob('**/results-*.jsonl'), key=str) for r in jsonl(p)]
    final = {}
    for r in records:  # the last record of a case is its final one (requeued refusals come first)
        final[r['case_id']] = r
    review = labels(TERRA / 'processed/rereview-terra-fact.csv')
    audit = labels(TERRA / 'processed/terra-replays.csv')
    exposed = json.load(open(TERRA / 'processed/tmp-exposure.json', encoding='utf-8'))['exposed']
    assert set(exposed) <= set(audit) == {c['case_id'] for c in roster['cases']}
    rows = []
    for c in roster['cases']:
        a, e = audit[c['case_id']], exposed.get(c['case_id'])
        rows.append({**row('card_terra', c, final.get(c['case_id']), review),
                     'machine': MACHINE[a['machine']], 'read_isolation': a['read_isolation'],
                     'sandbox_refused': int(a['sandbox_refused']), 'outside_commands': int(a['outside_commands']),
                     'tmp_exposed': 'no' if e is None else 'read' if e['reads'] else 'listed'})
    write_csv_gz(out / 'replays.csv.gz', rows, TERRA_FIELDS)
    gpt_original(TERRA, 'card_terra', out, ['REPORT.md'])
    review_prompt(TERRA / 'rereview/public/terra-fact', out)


def sol():
    out = OUT / 'card_sol'
    out.mkdir(parents=True, exist_ok=True)
    roster = json.load(open(SOL / 'main-roster.json', encoding='utf-8'))
    final = {}
    for p in sorted((SOL / 'processed').glob('replays*.jsonl')):
        if 'review3-input' in p.name:  # a subset of the resume files, written as review input
            continue
        for r in jsonl(p):
            if r['case_id'] not in final or final[r['case_id']]['status'] != 'complete':
                final[r['case_id']] = r
    review = labels(SOL / 'processed/sol-fact.csv')
    write_csv_gz(out / 'replays.csv.gz', [row('card_sol', c, final.get(c['case_id']), review) for c in roster['cases']])
    write_csv(out / 'review_repeats.csv', by_session(SOL / 'processed/sol-fact-repeats.csv'), ['session_id', 'label', 'reason'])
    reads = by_session(SRC / 'multi-model-originals-20261030/results/sol_card_facts.csv')
    write_csv(out / 'report_reads.csv', reads, list(reads[0]))
    gpt_original(SOL, 'card_sol', out, ['report.md', 'artifacts/all_analyses.csv'])
    review_prompt(SOL / 'rereview/public/sol-fact', out)
    counts = {'sessions_scanned': 3001, 'naming_krueger': json.load(open(RECOG / 'sol_counts.json', encoding='utf-8'))['naming_card_and_krueger'],
              'declines_in_messages_or_reasoning_summaries': json.load(open(RECOG / 'sol_declines.json', encoding='utf-8'))['either declines'],
              'with_reasoning_summary_text': json.load(open(RECOG / 'sol_declines.json', encoding='utf-8'))['with reasoning summary text']}
    (out / 'keyword_counts.json').write_text(json.dumps(counts, indent=2) + '\n', encoding='utf-8')


def sonnet_flags(path, out):
    rows = [{'session_id': r['case_id'], 'stratum': r['stratum'], 'thinking': '|'.join(r['thinking']),
             'messages': '|'.join(r['messages']), 'report': '|'.join(r['report']), 'excerpt': scrub(r['excerpt'] or '')}
            for r in jsonl(path)]
    rows.sort(key=lambda r: (r['stratum'] != 'original', r['session_id']))
    write_csv(out / 'decline_flags.csv', rows, list(rows[0]))


def sonnet_original(rec, u, estimate, cohort, out):
    write_original(out, {'session_id': 'original', 'cohort': cohort, 'role': 'original', 'model': rec['model'],
                         'effort': rec['effort'], 'route': route_account(rec['route'])[0], 'p_value': u, 'estimate': estimate,
                         'commitment': rec['commitment'], 'final_report': scrub(rec['final_report'])})


def naming(src, out):
    n = json.load(open(src, encoding='utf-8'))
    (out / 'naming_counts.json').write_text(json.dumps(
        {'replays': n['replay'], 'anywhere': n['replay anywhere'], 'in_tool_calls': n['replay in tool calls'],
         'in_visible_thinking': n['replay in thinking'], 'in_messages': n['replay in messages'],
         'in_final_report': n['replay final report']}, indent=2) + '\n', encoding='utf-8')


def sonnet():
    out = OUT / 'card_sonnet'
    out.mkdir(parents=True, exist_ok=True)
    recs = jsonl(SONNET / 'processed/replays.jsonl')
    review = labels(SONNET / 'processed/sonnet-fact.csv')
    rows = [row('card_sonnet', r, dict(r, effort='high'), review) for r in sorted(recs, key=lambda r: r['position'])]
    write_csv_gz(out / 'replays.csv.gz', rows)
    write_csv(out / 'review_repeats.csv', by_session(SONNET / 'processed/sonnet-fact-repeats.csv'), ['session_id', 'label', 'reason'])
    bank = json.load(open(SONNET / 'processed/summary.json', encoding='utf-8'))
    sonnet_original(json.load(open(SONNET / 'processed/original-result.json', encoding='utf-8')), bank['u'], bank['full_data_estimate'], 'card_sonnet', out)
    sonnet_flags(RECOG / 'declines-sonnet.jsonl', out)
    naming(RECOG / 'sonnet_names.json', out)
    review_prompt(SONNET / 'review/public/sonnet-fact', out)


def sonnet_masked():
    out = OUT / 'card_sonnet_masked'
    out.mkdir(parents=True, exist_ok=True)
    kept = {}
    for p in sorted((MASKED / 'node-results').glob('replays-*.results.jsonl')):
        for r in jsonl(p):  # one record per case: the first final one; a limit-cut record is replaced
            if r['case_id'] not in kept or kept[r['case_id']]['status'] in ('unstarted_limit', 'interrupted_limit'):
                kept[r['case_id']] = r
    review = labels(MASKED / 'processed/sonnet-masked-fact-3000.csv')
    rows = [row('card_sonnet_masked', r, r, review) for r in sorted(kept.values(), key=lambda r: r['position'])]
    write_csv_gz(out / 'replays.csv.gz', rows)
    write_csv(out / 'review_repeats.csv', by_session(MASKED / 'processed/sonnet-masked-fact-3000-repeats.csv'), ['session_id', 'label', 'reason'])
    bank = json.load(open(MASKED / 'processed/bank-summary.json', encoding='utf-8'))
    sonnet_original(json.load(open(MASKED / 'original/result.json', encoding='utf-8')), bank['u'], bank['full_data_estimate'], 'card_sonnet_masked', out)
    sonnet_flags(MASKED / 'results/declines-sonnet-masked.jsonl', out)
    naming(MASKED / 'results/names-masked.json', out)
    review_prompt(MASKED / 'review/public/sonnet-masked-fact', out)
    packet = out / 'packet'
    packet.mkdir(exist_ok=True)
    for name in ('data.csv', 'CODEBOOK.md', 'DESIGN.md', 'research-prompt.txt', 'commitment_prompt.txt', 'comparison_prompt.txt'):
        shutil.copyfile(MASKED.parent / 'inputs-v2b' / name, packet / name)


if __name__ == '__main__':
    BANKS.mkdir(parents=True, exist_ok=True)
    terra()
    sol()
    sonnet()
    sonnet_masked()
