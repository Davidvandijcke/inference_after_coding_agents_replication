#!/usr/bin/env python3
"""Redact the cluster's host paths and account ids from GPT-6-Luna's Card session records
(analysis-streams.jsonl.xz and the replay-requests extracts). Used by extract_records.py; no model calls.

Some agents listed the host's file system (rg over /, which, R.home()) or their workspace (ls -l), so their command
output and a few commands name the cluster's software tree and administrative folders, and directory listings show
the owner and group of the files, which the sandbox prints as the numeric ids of the account that ran the sessions.
Three rules apply to every string in every event of the streams and in every recorded tool call or user message of
the request extracts:
  - a line that names a folder under /usr/arcts/ (rg's "Permission denied" lines) becomes "<redacted host folder>";
  - a path under /sw/ (the cluster's software tree) becomes "<cluster-software-path>";
  - the owner and group fields of an `ls -l` line become "<owner>" and "<group>".
Records without such text keep their exact bytes; changed records are re-serialized in the source's JSON format.
The streams' eventstream_bytes and eventstream_sha256 describe each stream before redaction.
"""
import gzip
import json
import lzma
import re

ARCTS = '/usr/arcts/'
SW = re.compile(r'/sw/[^\s\'"`\\:;,()\[\]{}<>|]*')
# An `ls -l` line: mode, link count, owner, group, then size and date (month and day, or an ISO date).
LS = re.compile(r'([bcdlps-][rwxsStT-]{9}[.+@]? +\d+ +)[\w.$-]+( +)[\w.$-]+'
                r'(?= +[\d.,]+[A-Za-z]{0,3} +(?:[A-Z][a-z]{2} +\d{1,2} |\d{4}-\d{2}-\d{2} ))')
LINE, PATH, OWNER = '<redacted host folder>', '<cluster-software-path>', r'\1<owner>\2<group>'


def hit(s):
    return '/sw/' in s or ARCTS in s or LS.search(s) is not None


def text(s):
    if not hit(s):
        return s
    return '\n'.join(LINE if ARCTS in line else LS.sub(OWNER, SW.sub(PATH, line)) for line in s.split('\n'))


def value(x):
    if isinstance(x, str):
        return text(x)
    if isinstance(x, list):
        return [value(v) for v in x]
    if isinstance(x, dict):
        return {k: value(v) for k, v in x.items()}
    return x


def stream_line(line):
    """One session of analysis-streams.jsonl (compact JSON, sorted keys; its events compact JSON in source order)."""
    if not hit(line):
        return line
    r = json.loads(line)
    r['analysis_events_jsonl'] = '\n'.join(
        json.dumps(value(json.loads(e)), separators=(',', ':'), ensure_ascii=False) if hit(e) else e
        for e in r['analysis_events_jsonl'].split('\n'))
    return json.dumps(r, separators=(',', ':'), ensure_ascii=False, sort_keys=True)


def streams(source, target):
    with lzma.open(source, 'rt', encoding='utf-8') as f, lzma.open(target, 'wt', encoding='utf-8') as out:
        for line in f:
            out.write(stream_line(line.rstrip('\n')) + '\n')


def requests(source, target):
    """One replay-requests shard (JSON lines in json.dumps's default format)."""
    with gzip.open(source, 'rt', encoding='utf-8') as f, gzip.GzipFile(target, 'wb', mtime=0) as out:
        for line in f:
            line = line.rstrip('\n')
            if hit(line):
                line = json.dumps(value(json.loads(line)))
            out.write((line + '\n').encode('utf-8'))
