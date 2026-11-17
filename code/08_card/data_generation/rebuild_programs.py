#!/usr/bin/env python3
"""Rebuild the files a GPT-6-Luna Card session wrote, from the tool calls in its final model request.

The final request of a Codex session carries the whole conversation. Replaying its patches (apply_patch, each
applied in full or not at all, as in Codex) and its heredoc writes of analysis.R, up to the commitment prompt,
gives the session's final analysis.R. Used by extract_records.py for the original's workspace files and for the
replays' programs (data/agent_runs/card_luna/replay-programs.jsonl.gz), whose statuses are:
  ok               every patch applied
  patches_skipped  some patch did not apply (its context was not found) and was skipped whole
  compacted        the conversation was compacted, so earlier edits may be missing
  no_analysis_R    no analysis.R was written in the recorded calls

The replay extracts (data/agent_runs/card_luna/replay-requests/) were made on the cluster, read-only, from the
replays' closed case packs by the research repository's output/card-replay-estimates-20261029/code/gl_extract_calls.py.
No model calls.
Usage: python3 rebuild_programs.py <out.jsonl> (rebuilds every replay program from the package's extracts)
"""
import gzip
import json
import lzma
import re
import sys
from pathlib import Path

STR = re.compile(r'"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`')
HEREDOC = re.compile(r"(?:cat|tee)\s*(?:>\s*(\S*analysis\.R)\s*)?<<-?\s*['\"]?(\w+)['\"]?"
                     r"(?:\s*>\s*(\S*analysis\.R))?[^\n]*\n(.*?)\n\2[ \t]*(?:\n|$)", re.S)


class PatchError(Exception):
    pass


def items(request):
    """Tool calls and user messages of one model request, in order (as gl_extract_calls.py records them)."""
    out = []
    for i in request.get('input', []):
        t = i.get('type')
        if t in ('custom_tool_call', 'function_call'):
            out.append({'t': t, 'name': i.get('name'), 'input': i.get('input') or i.get('arguments')})
        elif t == 'message' and i.get('role') == 'user':
            out.append({'t': 'user', 'text': json.dumps(i.get('content'))[:300]})
        elif t == 'compaction' or (t and 'compact' in t):
            out.append({'t': 'compaction'})
    return out


def js_string(lit):
    if lit[0] == '`':
        return lit[1:-1].replace('\\`', '`').replace('\\$', '$').replace('\\\\', '\\')
    try:
        return json.loads(lit.replace("\\'", "'"))
    except ValueError:
        return None


def calls(code):
    """Patches and shell commands that mention analysis.R in one JavaScript tool block, in order."""
    out = []
    for m in STR.finditer(code):
        s = js_string(m.group(0))
        if s is None:
            continue
        if '*** Begin Patch' in s:
            out.append(('patch', s))
        elif 'analysis.R' in s:
            out.append(('cmd', s))
    return out


def find(lines, block, start):
    n = len(block)
    for norm in (lambda x: x, lambda x: x.rstrip(), lambda x: x.strip()):
        b = [norm(x) for x in block]
        for i in list(range(start, len(lines) - n + 1)) + list(range(0, min(start, len(lines) - n + 1))):
            if [norm(x) for x in lines[i:i + n]] == b:
                return i
    return -1


def apply_update(text, body):
    lines = text.split('\n')
    pos, hunks, cur, header = 0, [], None, None
    for ln in body:
        if ln.startswith('@@'):
            if cur:
                hunks.append((header, cur))
            header, cur = ln[2:].strip(), []
        elif not ln.startswith('*** End of File'):
            cur = (cur or []) + [ln]
    if cur:
        hunks.append((header, cur))
    for header, h in hunks:
        if header:
            j = find(lines, [header], pos)
            if j >= 0:
                pos = j + 1
        old = [x[1:] for x in h if x[:1] in (' ', '-') or x == '']
        new = [x[1:] for x in h if x[:1] in (' ', '+') or x == '']
        while old and new and old[-1] == '' and new[-1] == '':
            old.pop(); new.pop()
        if not old:
            lines[pos:pos] = new; pos += len(new); continue
        i = find(lines, old, pos)
        if i < 0:
            raise PatchError('context not found')
        lines[i:i + len(old)] = new
        pos = i + len(new)
    return '\n'.join(lines)


def workspace_path(p):
    return p if p.startswith('/') else '/workspace/' + (p[2:] if p.startswith('./') else p)


def apply_patch(files, patch):
    body = patch.split('\n')
    i = 0
    while i < len(body):
        m = re.match(r'\*\*\* (Add|Update|Delete) File: (.+)$', body[i])
        if not m:
            i += 1; continue
        op, path = m.group(1), workspace_path(m.group(2).strip())
        j, move = i + 1, None
        if j < len(body) and body[j].startswith('*** Move to: '):
            move = workspace_path(body[j][len('*** Move to: '):].strip()); j += 1
        k = j
        while k < len(body) and not re.match(r'\*\*\* (Add|Update|Delete) File: |\*\*\* End Patch', body[k]):
            k += 1
        chunk = body[j:k]
        try:
            if op == 'Add':
                files[path] = '\n'.join(x[1:] if x.startswith('+') else x for x in chunk)
            elif op == 'Delete':
                files.pop(path, None)
            else:
                if path not in files:
                    raise PatchError('update of unknown file')
                new = apply_update(files[path], chunk)
                if move:
                    files.pop(path)
                files[move or path] = new
        except PatchError:
            # Only the analysis script matters here; other files may have been written by shell commands.
            if path.endswith('/analysis.R') or (move or '').endswith('/analysis.R'):
                raise
            files.pop(path, None)
        i = k


def shell_write(files, cmd):
    for m in HEREDOC.finditer(cmd):
        target = m.group(1) or m.group(3)
        if target:
            files[workspace_path(target)] = m.group(4) + '\n'


def replay_edits(recorded):
    """Files written before the commitment prompt, and the number of patches that did not apply."""
    files, skipped = {}, 0
    for it in recorded:
        if it['t'] == 'user' and 'Commit to the analysis' in it.get('text', ''):
            break
        if it['t'] in ('custom_tool_call', 'function_call') and it.get('input'):
            for kind, s in calls(it['input']):
                before = dict(files)
                try:
                    apply_patch(files, s) if kind == 'patch' else shell_write(files, s)
                except PatchError:
                    files = before
                    skipped += 1
    return files, skipped


def workspace(recorded):
    return replay_edits(recorded)[0]


def rebuild(recorded):
    files, skipped = replay_edits(recorded)
    status = 'patches_skipped' if skipped else 'ok'
    if any(it['t'] == 'compaction' for it in recorded):
        status = 'compacted'
    text = files.get('/workspace/analysis.R')
    return text, (status if text is not None else 'no_analysis_R')


def rebuild_all(paths):
    """{replay id: (status, analysis.R text)} from the replay request extracts."""
    out = {}
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                text, status = rebuild(r['items'])
                out[r['job_id']] = (status, text)
    return out


def file_edits(streams):
    """Paths changed by each completed file edit, by session and event item id, from the event streams."""
    edits = {}
    with lzma.open(streams, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            d = edits.setdefault(r['case_id'], {})
            for event in r['analysis_events_jsonl'].splitlines():
                e = json.loads(event); item = e.get('item', {})
                if e.get('type') == 'item.completed' and item.get('type') == 'file_change':
                    d[item['id']] = [c['path'] for c in item.get('changes', [])]
    return edits


if __name__ == '__main__':
    runs = Path(__file__).resolve().parents[3] / 'data' / 'agent_runs' / 'card_luna'
    rebuilt = rebuild_all(sorted((runs / 'replay-requests').glob('shard-*.jsonl.gz')))
    with open(sys.argv[1], 'w', encoding='utf-8') as out:
        for k in sorted(rebuilt):
            out.write(json.dumps({'session_id': k, 'status': rebuilt[k][0], 'analysis_R': rebuilt[k][1]},
                                 ensure_ascii=False) + '\n')
