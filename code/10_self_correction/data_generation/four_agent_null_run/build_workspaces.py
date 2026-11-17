"""Rebuild the 100 original datasets from their seeds and write the case roster. No model calls.

Each dataset is generated with the main run's generate.py and compared with three records of the
main run: the file it served ($EXP/data/originals/main-XX.csv), the data_sha256 its runner recorded
for that original, and the frozen file hashes in main/MANIFEST.json (checked on the Mac). The run
stops before writing the roster if a single byte differs.

Usage (on Great Lakes, numpy on PYTHONPATH): python3 build_workspaces.py [--check-only]
"""
import glob
import hashlib
import json
import random
import sys

import common

sys.path.insert(0, str(common.FROZEN))
import generate  # noqa: E402

EXP = common.MAIN_RUN
SEED_BASE = 8_000_000
ORDER_SEED = 20261030


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main_run_hashes():
    """data_sha256 per slot from the main run's original records; the last record per case wins."""
    out = {}
    for p in sorted(glob.glob(str(EXP / "node-results" / "main-originals*.results.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            out[r["meta"]["slot"]] = {"case_id": r["case_id"], "data_sha256": r["data_sha256"]}
    return out


def main():
    data_dir = common.RUN / "data" / "originals"
    data_dir.mkdir(parents=True, exist_ok=True)
    recorded = main_run_hashes()
    rows, bad = [], []
    for slot in range(100):
        seed = SEED_BASE + slot
        name = f"main-{slot:02d}.csv"
        path = data_dir / name
        tmp = path.with_suffix(".tmp")
        generate.write_csv(generate.generate(seed, 2000), tmp)
        built = sha_bytes(tmp.read_bytes())
        tmp.replace(path)
        served = sha_bytes((EXP / "data" / "originals" / name).read_bytes())
        ok = built == served == recorded[slot]["data_sha256"]
        rows.append({"slot": slot, "seed": seed, "file": name, "sha256": built, "matches_served_file": built == served,
                     "matches_recorded_sha": built == recorded[slot]["data_sha256"],
                     "main_run_case_id": recorded[slot]["case_id"]})
        if not ok:
            bad.append(slot)
    frozen = {f: common.sha(common.FROZEN / f) for f in common.FROZEN_FILES}
    served_task = {f: common.sha(EXP / "source" / f) for f in common.FROZEN_FILES}
    check = {"datasets": len(rows), "identical_to_served_and_recorded": len(rows) - len(bad), "mismatched_slots": bad,
             "frozen_files": frozen, "frozen_files_match_main_run_source": frozen == served_task, "rows": rows}
    common.write_json(common.RUN / "WORKSPACE_CHECK.json", check)
    print(json.dumps({k: v for k, v in check.items() if k != "rows"}, indent=1))
    if bad or frozen != served_task:
        sys.exit("workspaces differ from the main run; no roster written")
    if "--check-only" in sys.argv:
        return
    cases = []
    for model in common.MODELS:
        order = list(range(100))
        random.Random(f"{ORDER_SEED}-{model}").shuffle(order)
        for position, slot in enumerate(order):
            cases.append({"case_id": hashlib.sha256(f"{model}:{slot}".encode()).hexdigest()[:32], "model": model,
                          "slot": slot, "seed": SEED_BASE + slot, "data": f"main-{slot:02d}.csv", "position": position})
    roster = common.RUN / "roster.json"
    if roster.exists():
        old = json.loads(roster.read_text())["cases"]
        if old != cases:
            sys.exit("a different roster already exists; it is sealed")
        print("roster already sealed and identical")
        return
    common.write_json(roster, {"order_seed": ORDER_SEED, "cases": cases})
    print(f"roster: {len(cases)} cases")


if __name__ == "__main__":
    main()
