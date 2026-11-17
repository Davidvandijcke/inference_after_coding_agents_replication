import tarfile, io, re, glob, json, collections
pat = re.compile(r"krueger", re.I)
n = collections.Counter(); ex = []
for t in sorted(glob.glob("out/*/cases-*.tar")):
    with tarfile.open(t) as outer:
        for m in outer.getmembers():
            if not m.name.endswith(".tar.gz") or "refused" in m.name:
                continue
            inner = tarfile.open(fileobj=io.BytesIO(outer.extractfile(m).read()))
            n["cases"] += 1
            where = set(); last_msg = ""
            for x in inner.getmembers():
                if not x.isfile() or not x.name.startswith("sessions/") or not x.name.endswith(".jsonl"):
                    continue
                for line in inner.extractfile(x).read().decode(errors="replace").splitlines():
                    if '"response_item"' not in line:
                        continue
                    try:
                        p = json.loads(line).get("payload") or {}
                    except ValueError:
                        continue
                    typ = p.get("type")
                    if typ == "message" and p.get("role") == "assistant":
                        txt = " ".join(c.get("text", "") for c in p.get("content") or [] if isinstance(c, dict))
                        last_msg = txt or last_msg
                        if pat.search(txt): where.add("messages")
                    elif typ in ("function_call", "custom_tool_call", "local_shell_call"):
                        if pat.search(json.dumps(p.get("arguments") or p.get("input") or p.get("action") or "")): where.add("commands or code it wrote")
                    elif typ in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
                        if pat.search(json.dumps(p.get("output") or "")): where.add("command output")
                    elif typ == "reasoning":
                        if pat.search(json.dumps(p.get("summary") or "")): where.add("reasoning summary")
            if where:
                n["mentions Krueger anywhere"] += 1
                for w in where: n["in " + w] += 1
            if pat.search(last_msg): n["final report mentions Krueger"] += 1
            if where and len(ex) < 3:
                ex.append(sorted(where))
print(dict(n)); print(ex)
