#!/usr/bin/env python3
"""Add the records kept with the session archives on the cluster to the four cohort folders.

Reads a local copy of the cluster run folders (first argument):
  sol/out/<job>/cases-*.tar                 GPT-6.1-Sol session archives (one .tar.gz per session)
  sonnet/node-archives/*.tar, sonnet/bank/bank-p.csv, sonnet/original/
  masked/node-archives/*.tar, masked/bank/bank-p.csv, masked/original/
and the research repository's output/ folder (second argument), whose Sol records name each completed
replay's final archive. Writes:
  data/agent_runs/card_sol/reports.jsonl.gz       final report of every completed Sol replay
  data/agent_runs/card_*/session_flags.csv        decline and naming keyword flags for every session archive
                                                  (Sol, Sonnet, masked Sonnet)
  data/agent_runs/card_sonnet{,_masked}/programs/original-analysis.R and original/ (report, specifications)
  data/intermediate/09_card_agents/sonnet{,_masked}_bank_p.csv
The decline rule is ../declines.py; naming is "krueger" anywhere, as in keyword_scans/. No model calls.
Usage: python3 extract_cluster_records.py <local copy> <research>/output
"""
import csv
import glob
import gzip
import io
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from declines import declines  # noqa: E402

COPY, SRC = Path(sys.argv[1]), Path(sys.argv[2])
ROOT = HERE.parents[2]
OUT = ROOT / 'data' / 'agent_runs'
BANKS = ROOT / 'data' / 'intermediate' / '09_card_agents'
NAME = re.compile(r'krueger', re.I)
WORKSPACE = re.compile(r'(/[^\s/()\[\]]+)+?/ws-[0-9a-f]+/')


def scrub(text):
    return WORKSPACE.sub('', text)


def inner_archives(pattern):
    """(outer tar name, member name, inner tarfile) for every session archive in the outer tars."""
    for outer_path in sorted(glob.glob(str(pattern))):
        with tarfile.open(outer_path) as outer:
            for m in outer.getmembers():
                if m.name.endswith('.tar.gz'):
                    with tarfile.open(fileobj=io.BytesIO(outer.extractfile(m).read()), mode='r:gz') as inner:
                        yield outer_path, m.name, inner


def patterns(text):
    return '|'.join(declines(text)[0])


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def sol():
    """Sol (Codex CLI): final report = last agent message of the analysis turn; flags over the session logs."""
    final = {}
    for p in sorted((SRC / 'card-sol-subscription-pilot-20261029/processed').glob('replays*.jsonl')):
        if 'review3-input' in p.name:
            continue
        for r in map(json.loads, open(p, encoding='utf-8')):
            if r['status'] == 'complete':
                final[r['case_id']] = r['archive']
    stratum = {c['case_id']: c['stratum'] for c in json.load(open(SRC / 'card-sol-subscription-pilot-20261029/main-roster.json', encoding='utf-8'))['cases']}
    reports, flags = {}, []
    for outer_path, member, inner in inner_archives(COPY / 'sol/out/*/cases-*.tar'):
        if 'refused' in member:
            continue
        case_id = member.split('-')[0]
        spec = f"out/{Path(outer_path).parent.name}/{Path(outer_path).name}:{member}"
        report, msgs, summ, tools, outputs = None, [], [], [], []
        for m in inner.getmembers():
            if not m.isfile():
                continue
            text = inner.extractfile(m).read().decode(errors='replace')
            if m.name.endswith('.jsonl') and 'analysis' in m.name and 'events' in m.name:
                for line in text.splitlines():
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get('type') == 'item.completed' and (e.get('item') or {}).get('type') == 'agent_message':
                        report = e['item'].get('text', '')
            elif m.name.startswith('sessions/') and m.name.endswith('.jsonl'):
                for line in text.splitlines():
                    if '"response_item"' not in line:
                        continue
                    try:
                        p = json.loads(line).get('payload') or {}
                    except ValueError:
                        continue
                    typ = p.get('type')
                    if typ == 'message' and p.get('role') == 'assistant':
                        msgs.append(' '.join(c.get('text', '') for c in p.get('content') or [] if isinstance(c, dict)))
                    elif typ == 'reasoning':
                        summ.append(' '.join(s.get('text', '') for s in p.get('summary') or [] if isinstance(s, dict)))
                    elif typ in ('function_call', 'custom_tool_call', 'local_shell_call'):
                        tools.append(json.dumps(p.get('arguments') or p.get('input') or p.get('action') or ''))
                    elif typ in ('function_call_output', 'custom_tool_call_output', 'local_shell_call_output'):
                        outputs.append(json.dumps(p.get('output') or ''))
        is_final = final.get(case_id) == spec
        if is_final:
            reports[case_id] = report
        flags.append({'session_id': case_id, 'archive': Path(member).name.removesuffix('.tar.gz'), 'final': is_final,
                      'stratum': stratum.get(case_id, ''), 'reasoning_summary_text': any(s.strip() for s in summ),
                      'decline_messages': patterns(' '.join(msgs)), 'decline_reasoning_summaries': patterns(' '.join(summ)),
                      'name_messages': any(NAME.search(x) for x in msgs), 'name_reasoning_summaries': any(NAME.search(x) for x in summ),
                      'name_tool_calls': any(NAME.search(x) for x in tools), 'name_tool_output': any(NAME.search(x) for x in outputs)})
    assert set(reports) == set(final), 'a completed Sol replay has no final archive'
    flags.sort(key=lambda r: (r['session_id'], r['archive']))
    write_csv(OUT / 'card_sol' / 'session_flags.csv', flags)
    with open(OUT / 'card_sol' / 'reports.jsonl.gz', 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as f:
        for case_id in sorted(reports):
            f.write((json.dumps({'session_id': case_id, 'stratum': stratum[case_id], 'final_report': scrub(reports[case_id] or '')},
                                ensure_ascii=False) + '\n').encode())


def sonnet(folder, cohort, results_glob):
    """Sonnet (Claude Code): flags over visible thinking, messages, tool calls and the final report."""
    stratum = {}
    for p in sorted(glob.glob(str(results_glob))):
        for line in open(p, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                if r.get('stratum'):
                    stratum[r['case_id']] = r['stratum']
    flags = []
    for outer_path, member, inner in inner_archives(COPY / folder / 'node-archives/*.tar'):
        ev = [x for x in inner.getnames() if x.endswith('logs/analysis.events.jsonl')]
        if not ev:
            continue
        case_id = ev[0].split('/')[0]
        think, texts, tools = [], [], []
        for line in inner.extractfile(ev[0]).read().decode(errors='replace').splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('type') == 'assistant':
                for b in (e.get('message') or {}).get('content') or []:
                    if not isinstance(b, dict):
                        continue
                    if b.get('type') == 'thinking':
                        think.append(b.get('thinking') or '')
                    elif b.get('type') == 'text':
                        texts.append(b.get('text') or '')
                    elif b.get('type') == 'tool_use':
                        tools.append(json.dumps(b.get('input') or {}))
        report = texts[-1] if texts else ''
        original = Path(outer_path).name.startswith('original')
        flags.append({'session_id': 'original' if original else case_id, 'stratum': 'original' if original else stratum[case_id],
                      'decline_thinking': patterns(' '.join(think)), 'decline_messages': patterns(' '.join(texts)),
                      'decline_report': patterns(report),
                      'name_thinking': any(NAME.search(x) for x in think), 'name_messages': any(NAME.search(x) for x in texts),
                      'name_tool_calls': any(NAME.search(x) for x in tools), 'name_report': bool(NAME.search(report))})
    flags.sort(key=lambda r: (r['stratum'] != 'original', r['session_id']))
    write_csv(OUT / cohort / 'session_flags.csv', flags)
    (OUT / cohort / 'programs').mkdir(exist_ok=True)
    shutil.copyfile(COPY / folder / 'original/emitted-analysis.R', OUT / cohort / 'programs' / 'original-analysis.R')
    shutil.copyfile(COPY / folder / 'bank/bank-p.csv', BANKS / f"{cohort.removeprefix('card_')}_bank_p.csv")


def sonnet_workspace(folder, cohort, files):
    dest = OUT / cohort / 'original'
    dest.mkdir(exist_ok=True)
    for name in files:
        (dest / Path(name).name).write_text(scrub((COPY / folder / 'original/workspace' / name).read_text(encoding='utf-8')), encoding='utf-8')


if __name__ == '__main__':
    sol()
    sonnet('sonnet', 'card_sonnet', COPY / 'sonnet/node-results/*.jsonl')
    sonnet_workspace('sonnet', 'card_sonnet', ['work/report.md', 'work/specification_log.csv'])
    sonnet('masked', 'card_sonnet_masked', SRC / 'card-sonnet-masked-20261104/pilot/node-results/*.results.jsonl')
    sonnet_workspace('masked', 'card_sonnet_masked', ['REPORT.md', 'output/all_specs.csv'])
