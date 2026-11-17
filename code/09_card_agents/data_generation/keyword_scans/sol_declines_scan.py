#!/usr/bin/env python3
"""Apply declines.py to GPT-6.1-Sol's Card session logs (assistant messages and reasoning summaries) and its original.
Run on Great Lakes from sol-card-main-gl with declines.py in the folder given as the first argument. No model calls."""
import collections, glob, io, json, sys, tarfile
sys.path.insert(0, sys.argv[1])
from declines import declines


def texts(lines):
    msgs, summ = [], []
    for line in lines:
        if '"response_item"' not in line:
            continue
        try:
            p = json.loads(line).get("payload") or {}
        except ValueError:
            continue
        if p.get("type") == "message" and p.get("role") == "assistant":
            msgs.append(" ".join(c.get("text", "") for c in p.get("content") or [] if isinstance(c, dict)))
        elif p.get("type") == "reasoning":
            summ.append(" ".join(s.get("text", "") for s in p.get("summary") or [] if isinstance(s, dict)))
    return msgs, summ


n = collections.Counter()
for t in sorted(glob.glob("out/*/cases-*.tar")):
    with tarfile.open(t) as outer:
        for m in outer.getmembers():
            if not m.name.endswith(".tar.gz") or "refused" in m.name:
                continue
            inner = tarfile.open(fileobj=io.BytesIO(outer.extractfile(m).read()))
            lines = []
            for x in inner.getmembers():
                if x.isfile() and x.name.startswith("sessions/") and x.name.endswith(".jsonl"):
                    lines += inner.extractfile(x).read().decode(errors="replace").splitlines()
            msgs, summ = texts(lines)
            n["cases"] += 1
            n["with reasoning summary text"] += any(s.strip() for s in summ)
            dm, ds = bool(declines(" ".join(msgs))[0]), bool(declines(" ".join(summ))[0])
            n["messages decline"] += dm; n["reasoning summaries decline"] += ds; n["either declines"] += dm or ds
print(json.dumps(dict(n)))
