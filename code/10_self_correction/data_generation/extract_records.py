#!/usr/bin/env python3
"""Build the compact agent records and synthetic datasets of module 10 (agents' self-corrections).

Writes
  data/agent_runs/multi_model_originals/   GPT-6.1-Sol, Claude Sonnet 5.5 and Claude Opus 5.5 originals on the
                                           100 open-ended null datasets (sessions.csv, reports.jsonl.gz, programs/)
                                           and the blind reads of all four agents' 400 reports (blind_reads.csv;
                                           GPT-6-Luna's reports are in open_ended_luna/, module 02)
  data/agent_runs/card_placebo_sol/        GPT-6.1-Sol on 200 full-size Card--Krueger placebo datasets
                                           (sessions.csv, reports.jsonl.gz, blind_reads.csv)
  data/agent_runs/sol_common_analysis/     GPT-6.1-Sol under the no-explore prompt: 60 originals on planted-effect
                                           datasets and 100 originals on fresh null datasets (planted_sessions.csv,
                                           reference_sessions.csv, reports.jsonl.gz, programs/)
  data/raw/pathways_sol_common/            the 160 generated datasets those 160 sessions analysed, regenerated from
                                           their seeds and checked against the hashes the sessions recorded
  data/raw/card_placebo/                   the 200 Card placebo datasets, regenerated the same way

Sources, in the research repository's output/ folder (first argument):
  multi-model-originals-20261030/archive/originals-*.tar.gz (emitted programs); results/per_original.csv and
      review2_labels.csv, and manuscript-rebuild-data/data/self-correction/pathways_reports.json, which the records
      built here are checked against
  card-conditional-size-20261102/   runtime/extension-records/ (runner records of the 160 confirmation sessions),
      frozen/ (generator and Card source), extension/results/confirmation.rows.csv (selected-analysis class);
      extension/results/blind_labels_combined400.csv, the merged reader export, is a check
  sol-recurrence-power-20261102/    audit/source/ (rosters, programs, generator), audit/membership.jsonl,
      followup_scoring/compact_runtime/reference/ and scored_records_reader.jsonl (reference sessions and labels)
  multiplicity-comparison-20261031/results/twin_panel_sol_rows.csv, twin_panel2_sol_rows.csv (checked against)
and in a staging copy of the cluster's run folders (second argument), fetched on 11 November 2026 with the same
relative paths as on the cluster:
  multi-model-originals-20261030/node-results/*.results.jsonl     runner records of the Sol, Sonnet and Opus originals
  multi-model-originals-20261030/review2/out/, private/key.json   raw blind-read outputs and the key to their ids
  card-conditional-size-20261102/lanes/sol-*/node-results/         runner records of the 40 pilot sessions
  card-conditional-size-20261102/review-clean-corrected-wave*/, extension/review-wave*/ (out/, private/key.json)
                                                                   raw blind reads of the 200 placebo reports
  multiplicity-comparison-20261031/twin-panel{,2}-sol/node-results/, read/review/out/, read/review/private/key.json
                                                                   runner records and raw blind reads, planted cohorts
No model calls. Usage: python3 extract_records.py <research>/output <staging>
"""
import ast
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

sys.dont_write_bytecode = True  # the generators are imported from the research repository, which stays untouched
SRC, STAGE = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / 'data' / 'agent_runs'
MMO = SRC / 'multi-model-originals-20261030'
CCS = SRC / 'card-conditional-size-20261102'
REC = SRC / 'sol-recurrence-power-20261102'
MULT = SRC / 'multiplicity-comparison-20261031'
DECL = ('model', 'null', 'population', 'outcome', 'sample_restrictions')
AGENTS = ('gpt-6.1-sol', 'claude-sonnet-5-5', 'claude-opus-5-5')  # GPT-6-Luna's originals are module 02's cohort
# Fields of a Card placebo blind read (data_generation/card_placebo/reader_prompt.txt)
READ_FIELDS = ['nominal_p', 'returned_p', 'returned_p_kind', 'outcome_construction', 'target_exposure', 'target_population',
               'status_missing_rules', 'restricted_population', 'heterogeneity_null', 'controls', 'scientific_model', 'exact_null',
               'tests_stated', 'own_correction', 'correction_tests', 'adjusted_p', 'survives', 'lead_decision',
               'alternative_headline_present', 'alternative_headline_summary', 'alternative_nominal_p', 'quotes']
SANDBOX = re.compile(r'(/private)?/tmp/ws-[0-9a-f]+/|/gpfs/\S*?/ws-[0-9a-f]+/|/scratch/\S*?/ws-[0-9a-f]+/')


def reset(folder):
    """Empty a generated folder, keeping its documentation (CODEBOOK.md, SOURCE.md)."""
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.iterdir():
        if p.suffix != '.md':
            shutil.rmtree(p) if p.is_dir() else p.unlink()


def scrub(text):
    """Agents' sandbox paths keep only the part inside the session's own workspace."""
    return SANDBOX.sub('', text) if isinstance(text, str) else text


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows({k: r.get(k, '') for k in fields} for r in rows)


def write_jsonl_gz(path, rows):
    with open(path, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as f:
        for r in rows:
            f.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n').encode())


def commitment(c):
    if isinstance(c, str):
        c = ast.literal_eval(c)  # some runner records hold the declaration as a dict repr
    return {f'commit_{k}': scrub((c or {}).get(k, '')) for k in DECL}


def route(rec):
    name, _, account = rec['route'].partition(':')
    return {'route': name, 'account': account}


def raw_reads(folder):
    """Blind-read outputs joined to the private key that maps each opaque id to its report and task."""
    key = json.loads((folder / 'private' / 'key.json').read_text(encoding='utf-8'))
    rows = []
    for f in sorted((folder / 'out').glob('results-*.jsonl')):
        for r in jsonl(f):
            assert r['ok'] and r.get('label')
            rows.append({**key[r['id']], 'read': r['read'], 'reviewer': r['reviewer'], **r['label']})
    return rows


def same(rows, reference, keys, fields):
    """Assert that rows rebuilt from the raw records equal the processed rows shipped before, field by field."""
    norm = lambda v: '' if v is None else str(v)
    ref = {tuple(norm(r[k]) for k in keys): r for r in reference}
    assert len(ref) == len(rows), (len(ref), len(rows))
    for r in rows:
        old = ref[tuple(norm(r[k]) for k in keys)]
        assert all(norm(r.get(f)) == norm(old.get(f)) for f in fields), (r, old)


def multi_model_originals():
    out = RUNS / 'multi_model_originals'
    reset(out)
    run = STAGE / 'multi-model-originals-20261030'
    kept = {}
    for f in sorted((run / 'node-results').glob('*.results.jsonl')):  # one record per case, a complete one first
        for r in jsonl(f):
            if r['case_id'] not in kept or r['status'] == 'complete':
                kept[r['case_id']] = r
    per = {(r['model'], int(r['slot'])): r for r in csv.DictReader(open(MMO / 'results' / 'per_original.csv', encoding='utf-8'))}
    sessions, reports = [], []
    for r in sorted(kept.values(), key=lambda r: (r['model'], r['slot'])):
        phases = r['phases'].values()
        usage = [p.get('usage') or {} for p in phases]
        cli = r['phases']['analysis'].get('cli_version')
        row = {'session_id': r['case_id'], 'cohort': 'multi_model_originals', 'role': 'original', 'model': r['model'], 'effort': r['effort'],
               'harness': f'Claude Code CLI {cli}' if cli else 'Codex CLI 0.165', **route(r), 'slot': r['slot'], 'seed': r['seed'],
               'data_sha256': r['data_sha256'], 'status': 'completed' if r['status'] == 'complete' else r['status'], 'attempt': r['attempt'],
               'u': '' if r.get('u') is None else repr(r['u']), **commitment(r.get('commitment')),
               'minutes': round((r['finished'] - r['started']) / 60, 2), 'n_commands': sum(p.get('n_commands') or 0 for p in phases),
               'tokens_input': sum(x.get('input_tokens', 0) + x.get('cache_read_input_tokens', 0) + x.get('cache_creation_input_tokens', 0) for x in usage),
               'tokens_output': sum(x.get('output_tokens', 0) for x in usage),
               'cost_usd_list': sum(p.get('total_cost_usd') or 0 for p in phases)}
        old = per[(r['model'], r['slot'])]
        usable = row['status'] == 'completed' and row['u'] != '' and row['commit_model'] != ''
        assert row['u'] == ('' if old['u'] == '' else repr(float(old['u']))) and str(usable) == old['usable']
        assert all(str(row[k]) == old[k] for k in ('minutes', 'n_commands', 'tokens_input', 'tokens_output'))
        sessions.append(row)
        reports.append({'model': r['model'], 'slot': r['slot'], 'final_report': r['final_report']})
    assert len(sessions) == 300
    write_csv(out / 'sessions.csv', sessions, list(sessions[0]))
    exported = json.load(open(SRC / 'manuscript-rebuild-data' / 'data' / 'self-correction' / 'pathways_reports.json', encoding='utf-8'))
    same(reports, [r for r in exported if r['model'] in AGENTS], ('model', 'slot'), ('final_report',))
    write_jsonl_gz(out / 'reports.jsonl.gz', [{**r, 'final_report': scrub(r['final_report'])} for r in reports])
    reads = sorted(raw_reads(run / 'review2'), key=lambda r: (r['model'], r['slot'], r['task'], r['read']))
    fields = ['model', 'slot', 'task', 'read', 'reviewer', 'category', 'tests_stated', 'other_significant', 'own_correction',
              'correction_tests', 'adjusted_p', 'survives', 'quote']
    same(reads, list(csv.DictReader(open(MMO / 'results' / 'review2_labels.csv', encoding='utf-8'))), ('model', 'slot', 'task', 'read'), fields[5:])
    write_csv(out / 'blind_reads.csv', reads, fields)
    for m in AGENTS:
        with tarfile.open(MMO / 'archive' / f'originals-{m}.tar.gz') as tar:
            for member in tar.getmembers():
                parts = member.name.split('/')
                if len(parts) == 5 and parts[3:] == ['workspace', 'score.py']:
                    dest = out / 'programs' / m / f'{parts[2]}.py'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())
    return len(sessions), len(reads)


def card_placebo_sol():
    out = RUNS / 'card_placebo_sol'
    reset(out)
    files = sorted((CCS / 'runtime' / 'extension-records').glob('*sol*.jsonl')) + \
        sorted((STAGE / 'card-conditional-size-20261102' / 'lanes').glob('sol-*/node-results/*.jsonl'))
    records = {}
    for f in files:
        for r in jsonl(f):
            if r.get('model') == 'gpt-6.1-sol' and r.get('status') == 'complete':
                records[r['case_id']] = r
    scored = {r['case_id']: r for r in csv.DictReader(open(CCS / 'extension' / 'results' / 'confirmation.rows.csv', encoding='utf-8'))}
    reads = []  # each Sol report was read once: pilot reads in four corrected waves, confirmation reads in five
    for cohort, pattern in (('pilot', 'review-clean-corrected-wave*'), ('confirmation', 'extension/review-wave*')):
        for folder in sorted((STAGE / 'card-conditional-size-20261102').glob(pattern)):
            for r in raw_reads(folder):
                if r['model'] == 'gpt-6.1-sol':
                    reads.append({**{k: scrub(v) for k, v in r.items()}, 'reader_cohort': cohort, 'reader_wave': folder.name[-1],
                                  'quotes': json.dumps(r['quotes'], ensure_ascii=False)})
    assert len(reads) == len({r['case_id'] for r in reads}) == 200
    exported = [r for r in csv.DictReader(open(CCS / 'extension' / 'results' / 'blind_labels_combined400.csv', encoding='utf-8')) if r['model'] == 'gpt-6.1-sol']
    same(reads, exported, ('case_id',), READ_FIELDS + ['reader_cohort', 'reviewer'])
    sessions, reports = [], []
    for lab in sorted(reads, key=lambda r: int(r['slot'])):
        cid, rec = lab['case_id'], records[lab['case_id']]
        assert abs(float(rec['u']) - float(scored[cid]['u'])) < 1e-12 and int(rec['seed']) == 117_020_000 + int(lab['slot'])
        sessions.append({'session_id': cid, 'cohort': 'card_placebo_sol', 'role': 'original', 'model': rec['model'], 'effort': rec['effort'],
                         'harness': 'Codex CLI 0.165', **route(rec), 'stage': lab['reader_cohort'], 'slot': int(lab['slot']),
                         'seed': int(rec['seed']), 'data_sha256': rec['data_sha256'], 'status': 'completed', 'u': repr(float(rec['u'])),
                         'selected_class': scored[cid]['class'], **commitment(rec['commitment'])})
        reports.append({'session_id': cid, 'final_report': scrub(rec['final_report'])})
    fields = ['session_id', 'cohort', 'role', 'model', 'effort', 'harness', 'route', 'account', 'stage', 'slot', 'seed', 'data_sha256',
              'status', 'u', 'selected_class'] + [f'commit_{k}' for k in DECL]
    write_csv(out / 'sessions.csv', sessions, fields)
    write_jsonl_gz(out / 'reports.jsonl.gz', reports)
    rows = sorted(reads, key=lambda r: int(r['slot']))
    fields = ['case_id', 'slot', 'reader_cohort', 'reader_wave', 'reviewer'] + READ_FIELDS
    with open(out / 'blind_reads.csv.gz', 'wb') as raw, gzip.GzipFile(filename='blind_reads.csv', fileobj=raw, mode='wb', mtime=0) as f:
        text = io.StringIO()
        w = csv.DictWriter(text, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in rows)
        f.write(text.getvalue().encode())
    # The 200 placebo datasets: the frozen generator permutes the state labels of the Card data file the agents
    # receive (frozen/source_card.csv, identical to the Card application's data.csv) with the session's seed.
    data_out = ROOT / 'data' / 'raw' / 'card_placebo'
    reset(data_out)
    gen = load_module('card_placebo_generator', HERE / 'card_placebo' / 'frozen' / 'generate.py')
    for row in sessions:
        name = f"placebo-{row['slot']:03d}.csv"
        gen.generate_csv(CCS / 'frozen' / 'source_card.csv', data_out / name, row['seed'])
        raw = (data_out / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['data_sha256'], name
        with open(data_out / (name + '.gz'), 'wb') as f, gzip.GzipFile(filename=name, fileobj=f, mode='wb', mtime=0) as g:
            g.write(raw)
        (data_out / name).unlink()
    return len(sessions), len(reports)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def procedure(code):
    """Standard errors and test reference of a program, as the procedure audit read them (sol_common_analysis/build_audit.py)."""
    robust = next((x for x in ('HC0', 'HC1', 'HC2', 'HC3') if x in code), None)
    if robust is None and re.search(r'1\s*-\s*h|leverage', code, re.I) and re.search(r't\.(?:sf|cdf)|stats\.t', code):
        robust = 'HC2'  # programs that spell out the HC2 leverage adjustment
    if re.search(r'use_t\s*=\s*True|t\.(?:sf|cdf)|stats\.t', code):
        return robust, 'student_t'
    if re.search(r'use_t\s*=\s*False|norm\.(?:sf|cdf)|stats\.norm', code):
        return robust, 'normal_z'
    return robust, 'unclear'


def sol_common_analysis():
    out = RUNS / 'sol_common_analysis'
    data_out = ROOT / 'data' / 'raw' / 'pathways_sol_common'
    reset(out)
    reset(data_out)
    gen = load_module('pathways_power_generator', HERE / 'sol_common_analysis' / 'frozen' / 'generate.py')
    membership = {(r['cohort'], r['slot']): r for r in jsonl(REC / 'audit' / 'membership.jsonl')}
    choices = load_module('code_choices', HERE / 'four_agent_null_run' / 'code_choices.py')
    planted, reports, datasets = [], [], []
    for cohort, rows in (('twin-panel-sol', 'twin_panel_sol_rows.csv'), ('twin-panel2-sol', 'twin_panel2_sol_rows.csv')):
        run = STAGE / 'multiplicity-comparison-20261031' / cohort
        recs = {r['slot']: r for f in sorted((run / 'node-results').glob('*.jsonl')) for r in jsonl(f)}
        roster = {c['case_id']: c for c in json.loads((run / 'roster.json').read_text(encoding='utf-8'))['cases']}
        reads = {r['slot']: r for r in raw_reads(run / 'read' / 'review')}
        coded = {int(r['slot']): r for r in csv.DictReader(open(MULT / 'results' / rows, encoding='utf-8'))}
        for slot in sorted(recs):
            r, read = recs[slot], reads[slot]
            prog = REC / 'audit' / 'source' / cohort / 'originals' / r['model'] / f'{slot:02d}' / 'workspace' / 'score.py'
            code = prog.read_text(encoding='utf-8', errors='replace')
            c = choices.code_one(code, r['commitment'])
            robust, reference = procedure(code)
            row = {'session_id': r['case_id'], 'cohort': cohort, 'role': 'original', 'model': r['model'], 'effort': r['effort'],
                   'harness': 'Codex CLI 0.165', **route(r), 'slot': slot, 'seed': r['seed'], 'tau': 0.28,
                   'data_file': roster[r['case_id']]['data'] + '.gz', 'data_sha256': r['data_sha256'],
                   'status': 'completed' if r['status'] == 'complete' else r['status'], 'u': repr(r['u']), **commitment(r['commitment']),
                   'outcome': c['outcome'], 'followup': c['followup'], 'restricted': c['restricted'], 'interaction': c['interaction'],
                   'group': ' '.join(c['group']), 'baseline_columns': len(c['baseline_columns']),
                   'robust_se': robust, 'reference_distribution': reference, 'reviewer': read['reviewer'],
                   **{k: read[k] for k in ('tests_stated', 'other_significant', 'own_correction', 'correction_tests', 'adjusted_p', 'survives')},
                   'read_quote': read['quote']}
            same([{**row, 'quote': row['read_quote']}], [coded[slot]], ('slot',),
                 ('outcome', 'followup', 'restricted', 'interaction', 'group', 'baseline_columns', 'tests_stated', 'own_correction',
                  'correction_tests', 'adjusted_p', 'survives', 'quote'))
            m = membership[(cohort, slot)]
            assert (robust, reference, r['case_id']) == (m['robust_se'], m['reference_distribution'], m['case_id']) and r['u'] == float(coded[slot]['u'])
            planted.append(row)
            reports.append({'session_id': r['case_id'], 'final_report': scrub(r['final_report'])})
            datasets.append((roster[r['case_id']]['data'], r['seed'], 0.28, r['data_sha256']))
            dest = out / 'programs' / f"{r['case_id']}.py"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prog, dest)
    write_csv(out / 'planted_sessions.csv', planted, list(planted[0]))
    ref_dir = REC / 'followup_scoring' / 'compact_runtime' / 'reference'
    recs = {r['case_id']: r for r in jsonl(ref_dir / 'records.jsonl')}
    roster = {c['case_id']: c for c in json.loads((ref_dir / 'roster.json').read_text(encoding='utf-8'))['cases']}
    scored = {r['case_id']: r for r in jsonl(REC / 'followup_scoring' / 'scored_records_reader.jsonl') if r['cohort'] == 'reference'}
    refs = []
    for cid in sorted(recs, key=lambda k: recs[k]['slot']):
        r, s = recs[cid], scored[cid]
        assert r['status'] == 'complete' and s['u'] == r['u']
        refs.append({'session_id': cid, 'cohort': 'reference-null', 'role': 'original', 'model': r['model'], 'effort': r['effort'],
                     'harness': 'Codex CLI 0.165', 'route': 'chatgpt_subscription', 'account': r['route'].split(':')[1],
                     'slot': r['slot'], 'seed': r['seed'], 'tau': 0.0, 'data_file': roster[cid]['data'] + '.gz',
                     'data_sha256': r['data_sha256'], 'status': 'completed', 'u': repr(r['u']), **commitment(r['commitment']),
                     'common_procedure': s['broad_common_procedure'], 'reader_label': s.get('reader_label') or '',
                     'reader_reason': s.get('reader_reason') or ''})
        reports.append({'session_id': cid, 'final_report': scrub(r['final_report'])})
        datasets.append((roster[cid]['data'], r['seed'], 0.0, r['data_sha256']))
        shutil.copyfile(ref_dir / 'scores' / f'{cid}.py', out / 'programs' / f'{cid}.py')
    write_csv(out / 'reference_sessions.csv', refs, list(refs[0]))
    write_jsonl_gz(out / 'reports.jsonl.gz', sorted(reports, key=lambda r: r['session_id']))
    for name, seed, tau, sha in datasets:
        tmp = data_out / name
        gen.write_csv(gen.generate(seed, 2000, tau), tmp)
        raw = tmp.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha, name
        with open(data_out / (name + '.gz'), 'wb') as f, gzip.GzipFile(filename=name, fileobj=f, mode='wb', mtime=0) as g:
            g.write(raw)
        tmp.unlink()
    return len(planted), len(refs), len(datasets)


if __name__ == '__main__':
    print('multi_model_originals sessions, reads:', multi_model_originals())
    print('card_placebo_sol sessions, reports:', card_placebo_sol())
    print('sol_common_analysis planted, reference, datasets:', sol_common_analysis())
