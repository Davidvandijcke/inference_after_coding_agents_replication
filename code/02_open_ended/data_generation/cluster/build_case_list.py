"""Concatenate the replay cases of the named originals list for the given arms into one
shuffled case list (fixed seed), and refuse to write it if the count is not the expected one.

Usage: python build_case_list.py ORIGINALS_LIST N_SELECT OUT.jsonl EXPECTED ARM [ARM...]
"""
import json
import random
import sys
from collections import Counter

import make_references as m
import runner

originals_list, n_select, out, expected = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
arms = sys.argv[5:]
cases = []
for c, _ in m.selected_originals(originals_list, n_select):
    for arm in arms:
        p = runner.EXP / "banks" / f"{c['case_id']}.{arm}.replays.jsonl"
        cases += [json.loads(l) for l in open(p) if l.strip()]
random.Random(20261027).shuffle(cases)
done = runner.load_done()
already = sum(c["case_id"] in done for c in cases)
print(json.dumps({"cases": len(cases), "already_done": already,
                  "by_arm_stratum": dict(Counter(f"{c['meta']['arm']}/{c['meta']['stratum']}" for c in cases)),
                  "unique_ids": len({c["case_id"] for c in cases})}))
if len(cases) != expected or len({c["case_id"] for c in cases}) != len(cases):
    sys.exit(f"expected {expected} unique cases")
with open(runner.EXP / out, "w") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")
print("wrote", out)
