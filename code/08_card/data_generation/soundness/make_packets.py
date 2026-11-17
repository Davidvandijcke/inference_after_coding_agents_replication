"""One review packet per classified Card replay: its committed declaration and its final
analysis.R with comments removed. No estimates, p-values, strata or match labels.
Usage: python3 make_packets.py  (from this folder)"""
import csv, gzip, json
from pathlib import Path

OUT = Path('..')
DECL = OUT / 'manuscript-rebuild-data/data/card-declarations.jsonl.gz'
JOIN = OUT / 'card-trace-coding-20261018-v1/extraction/PRIVATE_join.jsonl'
SCRIPTS = OUT / 'card-replay-estimates-20261029/processed/rebuilt-analysis-r.jsonl'
REPLAYS = OUT / 'card-mn-robustness-20261027/processed/card-replays.csv'
FIELDS = ('model', 'null', 'population', 'outcome', 'sample_restrictions')


def strip_comments(code):
    """Drop R comments (text after a # outside quotes) and the blank lines they leave."""
    out = []
    for line in code.split('\n'):
        quote, cut = None, len(line)
        for i, ch in enumerate(line):
            if quote:
                if ch == '\\':
                    continue
                if ch == quote and (i == 0 or line[i - 1] != '\\'):
                    quote = None
            elif ch in ('"', "'", '`'):
                quote = ch
            elif ch == '#':
                cut = i
                break
        kept = line[:cut].rstrip()
        if kept or (line.strip() == '' and out and out[-1] != ''):
            out.append(kept)
    return '\n'.join(out).strip() + '\n'


case_of = {json.loads(l)['coding_id']: json.loads(l)['case_id'] for l in open(JOIN)}
decl = {}
for line in gzip.open(DECL, 'rt'):
    r = json.loads(line)
    decl[case_of[r['coding_id']]] = r['declaration']
code = {json.loads(l)['replay_id']: json.loads(l)['analysis_R'] for l in open(SCRIPTS)}
classified = [r['replay_id'] for r in csv.DictReader(open(REPLAYS)) if r['verdict'] != 'unresolved']
with open('inputs/packets.jsonl', 'w') as f:
    for rid in sorted(classified):
        d = decl[rid]
        f.write(json.dumps({'replay_id': rid,
                            'declaration': '\n'.join(f'{k}: {d[k]}' for k in FIELDS),
                            'code': strip_comments(code[rid]) if code.get(rid) else None}) + '\n')
print(len(classified), 'packets;', sum(1 for r in classified if not code.get(r)), 'without a recovered script')
