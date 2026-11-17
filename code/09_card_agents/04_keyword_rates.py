#!/usr/bin/env python3
"""Decline and naming rates in the note to Table 8: a session declines when the rule in declines.py matches its
joined messages or reasoning, and names the study when "krueger" appears. Sol and Sonnet are read from per-session
flags and checked against the counts of the cluster scans; Luna is scanned here from its analysis turn.

Inputs:  data/agent_runs/card_{sol,sonnet,sonnet_masked}/session_flags.csv, card_sol/keyword_counts.json,
         card_sonnet{,_masked}/naming_counts.json and decline_flags.csv, card_luna/analysis-streams.jsonl.xz
Output:  data/derived/09_card_agents/keyword_rates.json
"""
import csv
import json
import lzma
import re
import sys
from pathlib import Path

from declines import declines

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'data' / 'agent_runs'
OUT = ROOT / 'data' / 'derived' / '09_card_agents'
NAME = re.compile(r'krueger', re.I)


def check(ok, message):
    if not ok:
        sys.exit(message)


def read(cohort, name):
    return list(csv.DictReader(open(RUNS / cohort / name, encoding='utf-8')))


def sonnet(cohort):
    flags = [r for r in read(cohort, 'session_flags.csv') if r['stratum'] != 'original']
    named = {place: sum(r[f'name_{place}'] == 'True' for r in flags) for place in ('thinking', 'messages', 'tool_calls', 'report')}
    n = {'sessions': len(flags), 'unit': 'replays', 'decline': sum(bool(r['decline_thinking'] or r['decline_messages']) for r in flags),
         'name': sum(any(r[f'name_{p}'] == 'True' for p in ('thinking', 'messages', 'tool_calls')) for r in flags)}
    before = {r['session_id']: r for r in read(cohort, 'decline_flags.csv')}
    check(all((r['decline_thinking'], r['decline_messages'], r['decline_report']) ==
              (before[r['session_id']]['thinking'], before[r['session_id']]['messages'], before[r['session_id']]['report'])
              for r in read(cohort, 'session_flags.csv')), f'{cohort}: decline flags differ from the cluster scan')
    counts = json.load(open(RUNS / cohort / 'naming_counts.json', encoding='utf-8'))
    check((counts['replays'], counts['anywhere'], counts['in_visible_thinking'], counts['in_messages'], counts['in_tool_calls'],
           counts['in_final_report']) == (n['sessions'], n['name'], named['thinking'], named['messages'], named['tool_calls'],
                                          named['report']), f'{cohort}: naming counts differ from the cluster scan')
    return n


def sol():
    flags = read('card_sol', 'session_flags.csv')
    places = ('messages', 'reasoning_summaries', 'tool_calls', 'tool_output')
    n = {'sessions': len(flags), 'unit': 'session archives', 'decline': sum(bool(r['decline_messages'] or r['decline_reasoning_summaries']) for r in flags),
         'name': sum(any(r[f'name_{p}'] == 'True' for p in places) for r in flags)}
    counts = json.load(open(RUNS / 'card_sol' / 'keyword_counts.json', encoding='utf-8'))
    check((counts['sessions_scanned'], counts['naming_krueger'], counts['declines_in_messages_or_reasoning_summaries'],
           counts['with_reasoning_summary_text']) == (n['sessions'], n['name'], n['decline'],
                                                      sum(r['reasoning_summary_text'] == 'True' for r in flags)),
          'card_sol: keyword counts differ from the cluster scan')
    return n


def luna():
    n = {'sessions': 0, 'unit': 'replays', 'decline': 0, 'name': 0}
    with lzma.open(RUNS / 'card_luna' / 'analysis-streams.jsonl.xz', 'rt', encoding='utf-8') as f:
        for line in f:
            env = json.loads(line)
            if env.get('is_original'):
                continue
            msgs, other = [], []
            for event in env['analysis_events_jsonl'].splitlines():
                try:
                    e = json.loads(event)
                except ValueError:
                    continue
                item = e.get('item') or {}
                if e.get('type') != 'item.completed':
                    continue
                if item.get('type') == 'agent_message':
                    msgs.append(item.get('text', ''))
                elif item.get('type') == 'command_execution':
                    other.append((item.get('command') or '') + '\n' + (item.get('aggregated_output') or ''))
            n['sessions'] += 1
            n['decline'] += bool(declines(' '.join(msgs))[0])
            n['name'] += any(NAME.search(x) for x in msgs + other)
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rates = {'card_luna': luna(), 'card_sol': sol(),
             'card_sonnet': sonnet('card_sonnet'), 'card_sonnet_masked': sonnet('card_sonnet_masked')}
    for r in rates.values():
        r['decline_share'] = r['decline'] / r['sessions']
        r['name_share'] = r['name'] / r['sessions']
    (OUT / 'keyword_rates.json').write_text(json.dumps(rates, indent=2) + '\n', encoding='utf-8')
    for c, r in rates.items():
        print(f"{c:<19} decline {r['decline']:>5} of {r['sessions']} {r['unit']} ({100 * r['decline_share']:.1f}%)  "
              f"name {r['name']:>4} ({100 * r['name_share']:.1f}%)")


if __name__ == '__main__':
    main()
