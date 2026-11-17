#!/usr/bin/env python3
"""Write cases.jsonl for the m-value reference run: 20 full-data sessions for each of three briefs (29 October).

Every case gets the whole 46-country file in its own seeded row order (as the three originals did), the frozen
research prompt or one of its two variants, and kind "reference" (analysis phase only). Case ids are opaque 32-hex
strings; the launch order is shuffled with a fixed seed so the briefs interleave across nodes.
Usage: python make_cases.py SOURCE_CSV > cases.jsonl
"""
import hashlib
import json
import random
import sys

SEED = "issp-mvalue-reference-20261029"
BRIEFS = {"pro_team": "research_prompt.txt", "anti_team": "research_prompt_anti.txt",
          "nonpartisan": "research_prompt_nonpartisan.txt"}
cases = []
for brief, prompt in BRIEFS.items():
    for i in range(1, 21):
        cid = hashlib.sha256(f"{SEED}-{brief}-{i}".encode()).hexdigest()[:32]
        cases.append({"case_id": cid, "kind": "reference", "prompt_file": prompt,
                      "countries": {"source": sys.argv[1], "seed": SEED, "bank_index": f"{brief}-{i}", "k": None},
                      "meta": {"brief": brief, "session": i}})
random.Random(SEED).shuffle(cases)
for c in cases:
    print(json.dumps(c, sort_keys=True))
