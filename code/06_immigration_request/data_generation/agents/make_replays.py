"""Build the country-subsample bank for the ISSP original and select tail/body replays (no model calls).

The original is the first candidate, in slot order, with a usable commitment and an analysis_final.py whose
p-value on its own data.csv is defined. Each bank member keeps K whole countries of the 46, written exactly as the
runner writes a replay's data.csv (runner.write_countries); the original's frozen analysis_final.py is run on it in
a network-isolated sandbox. Members split into tail (p <= u) and body (p > u); the replays are a uniformly random
draw without replacement from each. Bank files exist only in node-local storage while being scored; replay cases
carry the country spec and the runner rebuilds the data on the compute node.

Usage: python make_replays.py --originals-list originals.jsonl --tail 500 --body 500 --bank 4000 --max-bank 16000
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import secrets
import shutil
from pathlib import Path

import runner

K = 16


def selected_original(list_name):
    cases = [json.loads(l) for l in open(runner.EXP / list_name)]
    results = runner.load_results()
    for c in sorted(cases, key=lambda c: c["meta"]["slot"]):
        r = results.get(c["case_id"])
        if r is not None and r.get("commitment") and r.get("u") is not None:
            return c, r
    raise SystemExit("no usable original yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals-list", default="originals.jsonl")
    ap.add_argument("--bank", type=int, default=4000)
    ap.add_argument("--max-bank", type=int, default=16000)
    ap.add_argument("--tail", type=int, default=500)
    ap.add_argument("--body", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    case, res = selected_original(a.originals_list)
    cid, seed, u = case["case_id"], case["meta"]["seed"], res["u"]
    source = case["countries"]["source"]
    workspace = runner.EXP / "originals" / cid / "workspace"
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"bank-{cid}"
    tmp.mkdir(parents=True, exist_ok=True)

    def spec(j):
        return {"source": source, "seed": seed, "bank_index": j, "k": K}

    def score(j):
        path = tmp / f"sub-{j}.csv"
        runner.write_countries(spec(j), path)
        r = runner.score_dataset(workspace, str(path), tmp / f"run-{j}")
        shutil.rmtree(tmp / f"run-{j}", ignore_errors=True)
        path.unlink()
        return j, runner.p_from(r["result"]), (r["stderr"][-300:] if runner.p_from(r["result"]) is None else "")

    scores, errors = {}, {}
    J = a.bank
    while True:
        todo = [j for j in range(J) if j not in scores]
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            for j, p, err in pool.map(score, todo):
                scores[j] = p
                if p is None:
                    errors[j] = err
        tail = [j for j, p in scores.items() if p is not None and p <= u]
        body = [j for j, p in scores.items() if p is not None and p > u]
        if (len(tail) >= a.tail and len(body) >= a.body) or J >= a.max_bank:
            break
        J = min(a.max_bank, 2 * J)

    pick = random.Random(f"{seed}-select")
    tail, body = sorted(tail), sorted(body)
    pick.shuffle(tail)
    pick.shuffle(body)
    replay_cases = []
    for stratum, js in (("tail", tail[:a.tail]), ("body", body[:a.body])):
        for j in js:
            replay_cases.append({"case_id": secrets.token_hex(16), "kind": "replay", "countries": spec(j),
                                 "target_commitment": res["commitment"],
                                 "meta": {"original": cid, "stratum": stratum, "bank_index": j,
                                          "subsample_p": scores[j]}})
    random.Random(f"{seed}-order").shuffle(replay_cases)  # launch order mixes the strata
    runner.write_json(runner.EXP / "banks" / f"{cid}.json",
                      {"original": cid, "u": u, "bank_size": J, "k": K, "n_countries": 46,
                       "n_scored": sum(p is not None for p in scores.values()), "n_tail": len(tail),
                       "n_body": len(body), "scores": scores, "errors": errors})
    with open(runner.EXP / "banks" / f"{cid}.replays.jsonl", "w") as f:
        for c in replay_cases:
            f.write(json.dumps(c) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"original {cid}: u {u:.4g}, bank {J}, tail {len(tail)}, body {len(body)}, errors {len(errors)}, "
          f"replays {len(replay_cases)}")


if __name__ == "__main__":
    main()
