#!/usr/bin/env python3
"""Count Sonnet Card sessions that name Krueger, by place (visible thinking, messages, final report, tool calls:
commands and file contents written). Same pattern as the Luna and Sol scans ("krueger"); it replaces
sonnet_sessions_scan.py's naming count, whose pattern also matched "nber" inside "number". Run on Great Lakes from
sonnet-card-main-gl. No model calls."""
import collections, glob, io, json, re, tarfile
pat = re.compile(r"krueger", re.I)
n = collections.Counter()
for t in sorted(glob.glob("node-archives/*.tar")):
    kind = "original" if t.split("/")[-1].startswith("original") else "replay"
    with tarfile.open(t) as outer:
        for m in outer.getmembers():
            if not m.name.endswith(".tar.gz"):
                continue
            inner = tarfile.open(fileobj=io.BytesIO(outer.extractfile(m).read()))
            ev = [x for x in inner.getnames() if x.endswith("logs/analysis.events.jsonl")]
            if not ev:
                continue
            think, texts, tools = [], [], []
            for line in inner.extractfile(ev[0]).read().decode(errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "assistant":
                    for b in (e.get("message") or {}).get("content") or []:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "thinking":
                            think.append(b.get("thinking") or "")
                        elif b.get("type") == "text":
                            texts.append(b.get("text") or "")
                        elif b.get("type") == "tool_use":
                            tools.append(json.dumps(b.get("input") or {}))
            n[kind] += 1
            where = {"thinking": think, "messages": texts, "tool calls": tools}
            hit = {w for w, xs in where.items() if any(pat.search(x) for x in xs)}
            if hit:
                n[kind + " anywhere"] += 1
            for w in hit:
                n[kind + " in " + w] += 1
            if texts and pat.search(texts[-1]):
                n[kind + " final report"] += 1
print(json.dumps(dict(n)))
