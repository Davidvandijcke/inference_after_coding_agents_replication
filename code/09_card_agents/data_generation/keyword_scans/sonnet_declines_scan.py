#!/usr/bin/env python3
"""Apply declines.py to every Sonnet Card session (analysis phase: visible thinking, messages, final report).
Run on Great Lakes from sonnet-card-main-gl with declines.py beside it. Writes declines-sonnet.jsonl. No model calls."""
import glob, io, json, random, sys, tarfile, collections
sys.path.insert(0, sys.argv[1])
from declines import declines
strata = {}
for f in glob.glob("node-results/replays-*.jsonl"):
    for l in open(f):
        if l.strip():
            r = json.loads(l); strata[r["case_id"]] = r["stratum"]
out = []
for t in sorted(glob.glob("node-archives/*.tar")):
    with tarfile.open(t) as outer:
        for m in outer.getmembers():
            if not m.name.endswith(".tar.gz"):
                continue
            inner = tarfile.open(fileobj=io.BytesIO(outer.extractfile(m).read()))
            ev = [x for x in inner.getnames() if x.endswith("logs/analysis.events.jsonl")]
            if not ev:
                continue
            cid = ev[0].split("/")[0]
            think, texts = [], []
            for line in inner.extractfile(ev[0]).read().decode(errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "assistant":
                    for b in (e.get("message") or {}).get("content") or []:
                        if isinstance(b, dict) and b.get("type") == "thinking":
                            think.append(b.get("thinking") or "")
                        elif isinstance(b, dict) and b.get("type") == "text":
                            texts.append(b.get("text") or "")
            report = texts[-1] if texts else ""
            h_t, x_t = declines(" ".join(think)); h_m, x_m = declines(" ".join(texts)); h_r, x_r = declines(report)
            out.append({"case_id": cid, "stratum": strata.get(cid, "original"), "thinking": h_t, "messages": h_m, "report": h_r,
                        "excerpt": x_r or x_m or x_t, "report_head": report[:500]})
with open(sys.argv[2], "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n = collections.Counter()
for r in out:
    s = r["stratum"]; n[s] += 1
    if r["thinking"]: n[s + " thinking"] += 1
    if r["messages"]: n[s + " messages"] += 1
    if r["report"]: n[s + " report"] += 1
    if r["thinking"] or r["messages"]: n[s + " thinking or messages"] += 1
print(dict(n))
print("patterns among replays:", collections.Counter(p for r in out if r["stratum"] != "original" for p in set(r["thinking"] + r["messages"])))
