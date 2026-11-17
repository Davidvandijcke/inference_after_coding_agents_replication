"""Keyword scan of Card replay reports for a named multiplicity method. No model calls.

Opens each case archive (a tar of per-case .tar.gz files, or loose .tar.gz files), takes the last agent
message of the analysis phase, and counts reports naming a correction method and reports saying the
result was selected from a search. Same regexes as correction_words.py and the Terra scan on the Mac.

Usage: python3 card_correction_scan.py LABEL PATH [PATH ...]
"""
import glob
import io
import json
import re
import sys
import tarfile

NAMED = re.compile(r"bonferroni|\bholm\b|benjamini|\bBH\b|\bFDR\b|false[- ]discovery|max-?T\b|westfall|family-?wise|multiple[- ](testing|comparison|hypothes|test)|multiplicity", re.I)
SOFT = re.compile(r"specification search|many specifications|several specifications|searched|selected (after|because)|chosen (after|because)|cherry|p-hack|garden", re.I)


def last_message(tar):
    txt = None
    for m in tar.getmembers():
        if m.name.endswith(".jsonl") and "analysis" in m.name and "events" in m.name:
            for line in tar.extractfile(m).read().decode(errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message":
                    txt = e["item"].get("text", "")
    return txt


def archives(paths):
    for p in paths:
        for f in sorted(glob.glob(p)):
            if f.endswith(".tar.gz"):
                with tarfile.open(f) as t:
                    yield last_message(t)
            elif f.endswith(".tar"):
                with tarfile.open(f) as outer:
                    for m in outer.getmembers():
                        if m.name.endswith(".tar.gz"):
                            with tarfile.open(fileobj=io.BytesIO(outer.extractfile(m).read()), mode="r:gz") as t:
                                yield last_message(t)


def main():
    label, paths = sys.argv[1], sys.argv[2:]
    n = hit = soft = 0
    examples = []
    for txt in archives(paths):
        if txt is None:
            continue
        n += 1
        if NAMED.search(txt):
            hit += 1
            if len(examples) < 3:
                m = NAMED.search(txt)
                examples.append(txt[max(0, m.start() - 150):m.end() + 120].replace("\n", " "))
        if SOFT.search(txt):
            soft += 1
    print(json.dumps({"label": label, "reports": n, "named_method": hit, "search_words": soft}))
    for x in examples:
        print("  -", x)


if __name__ == "__main__":
    main()
