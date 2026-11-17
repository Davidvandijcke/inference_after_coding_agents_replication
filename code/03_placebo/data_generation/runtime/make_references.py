"""Build a reference bank and replay cases for one original, in one of two arms.

fresh      Reference datasets are new draws from the generator with the original's
           size (seed = FRESH_NAMESPACE + slot * 1_000_000 + j). Replays are drawn
           uniformly without replacement within the tail (p <= u) and body (p > u).
subsample  Reference datasets are subsamples of m participants of the original's
           data, without replacement in random order. Replays are drawn by the
           stratified design: each draw picks a p-value bin uniformly (NB1
           equal-width bins on (0, u] for the tail, NB0 on (u, 1] for the body) and
           then an unused bank member of that bin at random. The bank grows until
           every bin can supply its draws.
           With --draw raw the replays are instead drawn uniformly without
           replacement within the tail and body, exactly as the paper's raw
           estimator describes (arm label "subsample-raw"); the bin of every draw
           is still recorded so that a post-stratified estimate can be computed.

Every reference dataset is scored with the original's frozen score.py in a
network-isolated sandbox and exists only in node-local storage while scored.
Replay cases carry a spec from which the runner regenerates the dataset byte for
byte. Outputs: banks/<cid>.<arm>.json and banks/<cid>.<arm>.replays.jsonl.

Usage (one original per array task):
  python make_references.py --arm fresh|subsample --task N [--originals-list LIST]
      [--n-select 8] [--tail 40] [--body 200] [--m 500] [--nb0 10] [--nb1 4]
      [--draw standardized|raw]
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

import generate
import runner

FRESH_NAMESPACE = 8_000_000_000


def case_id(original, arm, stratum, j):
    return hashlib.sha256(f"{original}:{arm}:{stratum}:{j}".encode()).hexdigest()[:32]


def selected_originals(list_name, n_select):
    cases = [json.loads(l) for l in open(runner.EXP / list_name) if l.strip()]
    results = runner.load_results()
    chosen = []
    for c in sorted(cases, key=lambda c: c["meta"]["slot"]):
        r = results.get(c["case_id"])
        if r and r.get("commitment") and r.get("u") is not None:
            chosen.append((c, r))
        if len(chosen) == n_select:
            break
    return chosen


def bin_of(p, lo, hi, nb):
    """Bin index of p in (lo, hi] split into nb equal-width bins; p == lo goes to bin 0."""
    return min(nb - 1, max(0, math.ceil((p - lo) / (hi - lo) * nb) - 1))


def ks_uniform(ps, lo=0.0, hi=1.0):
    """One-sample KS distance of ps from uniform on (lo, hi], with the asymptotic p-value."""
    x = sorted((p - lo) / (hi - lo) for p in ps)
    n = len(x)
    if n == 0:
        return None, None
    d = max(max((i + 1) / n - v, v - i / n) for i, v in enumerate(x))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    pval = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(1.0, max(0.0, pval))


def fresh_seed(slot, j):
    return FRESH_NAMESPACE + slot * 1_000_000 + j


def read_rows(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("fresh", "subsample"), required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--originals-list", default="originals.jsonl")
    ap.add_argument("--n-select", type=int, default=8)
    ap.add_argument("--tail", type=int, default=40)
    ap.add_argument("--body", type=int, default=200)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--nb0", type=int, default=10)
    ap.add_argument("--nb1", type=int, default=4)
    ap.add_argument("--draw", choices=("standardized", "raw"), default="standardized",
                    help="subsample arm only: stratified-by-bin draws (default) or uniform draws within stratum")
    ap.add_argument("--start-bank", type=int, default=None)
    ap.add_argument("--max-bank", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    if a.arm == "fresh" and a.draw != "standardized":
        raise SystemExit("--draw applies to the subsample arm only")
    arm_name = "subsample-raw" if (a.arm == "subsample" and a.draw == "raw") else a.arm
    start = a.start_bank or (4000 if a.arm == "fresh" else 2000)
    cap = a.max_bank or (128_000 if a.arm == "fresh" else 32_000)

    chosen = selected_originals(a.originals_list, a.n_select)
    if a.task >= len(chosen):
        return
    case, res = chosen[a.task]
    cid, slot, seed, u = case["case_id"], case["meta"]["slot"], case["meta"]["seed"], res["u"]
    n_fresh = case["meta"]["n"]
    if a.m == 0:
        a.m = int(0.10 * n_fresh + 0.5)
    workspace = runner.EXP / "originals" / cid / "workspace"
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"bank-{arm_name}-{cid}"
    tmp.mkdir(parents=True, exist_ok=True)
    if a.arm == "subsample":
        header, rows = read_rows(case["data_path"])
        n = len(rows)
        if a.m >= n:
            raise SystemExit(f"m={a.m} must be smaller than n={n}")

    def write_reference(j, path):
        if a.arm == "fresh":
            generate.write_csv(generate.generate(fresh_seed(slot, j), n_fresh), path)
        else:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows[i] for i in runner.subsample_indices(seed, j, n, a.m))

    def score(j):
        path = tmp / f"ref-{j}.csv"
        write_reference(j, path)
        r = runner.score_dataset(workspace, str(path), tmp / f"run-{j}")
        shutil.rmtree(tmp / f"run-{j}", ignore_errors=True)
        path.unlink()
        return j, runner.p_from(r["result"])

    def spec(j):
        if a.arm == "fresh":
            return {"fresh": {"seed": fresh_seed(slot, j), "n": n_fresh}}
        return {"subsample": {"source": case["data_path"], "seed": seed, "bank_index": j, "m": a.m}}

    scores = {}
    J = start

    def score_upto(J):
        todo = [j for j in range(J) if j not in scores]
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            scores.update(pool.map(score, todo))

    draws = []  # (stratum, bin, j) in draw order
    bank = {"original": cid, "slot": slot, "arm": arm_name, "draw": a.draw, "u": u, "seed": seed,
            "tail_requested": a.tail, "body_requested": a.body}

    def bin_key(p):
        return ("tail", bin_of(p, 0.0, u, a.nb1)) if p <= u else ("body", bin_of(p, u, 1.0, a.nb0))

    if a.arm == "fresh":
        while True:
            score_upto(J)
            tail = sorted(j for j, p in scores.items() if p is not None and p <= u)
            body = sorted(j for j, p in scores.items() if p is not None and p > u)
            if (len(tail) >= a.tail and len(body) >= a.body) or J >= cap:
                break
            J = min(cap, 2 * J)
        pick = random.Random(f"{seed}-fresh-select")
        pick.shuffle(tail)
        pick.shuffle(body)
        draws = [("tail", None, j) for j in tail[:a.tail]] + [("body", None, j) for j in body[:a.body]]
        valid = [p for p in scores.values() if p is not None]
        d, pv = ks_uniform(valid)
        bank.update(n=n_fresh, bank_size=J, n_scored=len(valid), n_tail=len(tail), n_body=len(body),
                    tail_shortfall=max(0, a.tail - len(tail)), body_shortfall=max(0, a.body - len(body)),
                    bank_ks=d, bank_ks_p=pv, bank_tail_frac=len(tail) / max(1, len(valid)))
    elif a.draw == "raw":
        while True:
            score_upto(J)
            tail = sorted(j for j, p in scores.items() if p is not None and p <= u)
            body = sorted(j for j, p in scores.items() if p is not None and p > u)
            if (len(tail) >= a.tail and len(body) >= a.body) or J >= cap:
                break
            J = min(cap, 2 * J)
        pick = random.Random(f"{seed}-subsample-raw-select")
        pick.shuffle(tail)
        pick.shuffle(body)
        draws = [("tail", bin_key(scores[j])[1], j) for j in tail[:a.tail]] + \
                [("body", bin_key(scores[j])[1], j) for j in body[:a.body]]
        members = {}
        for j, p in sorted(scores.items()):
            if p is not None:
                members.setdefault(bin_key(p), []).append(j)
        demand = {}
        for s, b, j in draws:
            demand[(s, b)] = demand.get((s, b), 0) + 1
        valid = [p for p in scores.values() if p is not None]
        body_ps = [p for p in valid if p > u]
        d, pv = ks_uniform(body_ps, u, 1.0) if body_ps else (None, None)
        bank.update(m=a.m, n=n, bank_size=J, n_scored=len(valid), n_tail=len(tail), n_body=len(body_ps),
                    nb0=a.nb0, nb1=a.nb1,
                    tail_shortfall=max(0, a.tail - len(tail)), body_shortfall=max(0, a.body - len(body)),
                    bins={f"{k[0]}-{k[1]}": {"bank_members": len(members.get(k, [])), "demand": demand.get(k, 0)}
                          for k in sorted(set(members) | set(demand))},
                    bank_tail_frac_over_u=(len(tail) / max(1, len(valid))) / u if u > 0 else None,
                    body_ks=d, body_ks_p=pv)
    else:
        pick = random.Random(f"{seed}-subsample-select")
        seq = [("tail", pick.randrange(a.nb1)) for _ in range(a.tail)] + \
              [("body", pick.randrange(a.nb0)) for _ in range(a.body)]
        demand = {}
        for s, b in seq:
            demand[(s, b)] = demand.get((s, b), 0) + 1
        while True:
            score_upto(J)
            members = {}
            for j, p in sorted(scores.items()):
                if p is None:
                    continue
                members.setdefault(bin_key(p), []).append(j)
            short = {k: v - len(members.get(k, [])) for k, v in demand.items() if v > len(members.get(k, []))}
            if not short or J >= cap:
                break
            J = min(cap, 2 * J)
        pools = {}
        for key, js in members.items():
            js = list(js)
            random.Random(f"{seed}-subsample-assign-{key[0]}-{key[1]}").shuffle(js)
            pools[key] = js
        for s, b in seq:
            pool = pools.get((s, b), [])
            if pool:
                draws.append((s, b, pool.pop()))
        valid = [p for p in scores.values() if p is not None]
        body_ps = [p for p in valid if p > u]
        d, pv = ks_uniform(body_ps, u, 1.0) if body_ps else (None, None)
        n_tail = sum(1 for p in valid if p <= u)
        bank.update(m=a.m, n=n, bank_size=J, n_scored=len(valid), n_tail=n_tail, n_body=len(body_ps),
                    nb0=a.nb0, nb1=a.nb1,
                    bins={f"{k[0]}-{k[1]}": {"bank_members": len(members.get(k, [])), "demand": demand.get(k, 0)}
                          for k in sorted(set(members) | set(demand))},
                    shortfall={f"{k[0]}-{k[1]}": v for k, v in short.items()},
                    bank_tail_frac_over_u=(n_tail / max(1, len(valid))) / u if u > 0 else None,
                    body_ks=d, body_ks_p=pv)

    bank["scores"] = {str(j): p for j, p in sorted(scores.items())}
    bank["draws"] = [{"stratum": s, "bin": b, "bank_index": j} for s, b, j in draws]
    replay_cases = []
    for s, b, j in draws:
        meta = {"original": cid, "slot": slot, "experiment": case["meta"]["experiment"], "arm": arm_name,
                "stratum": s, "bin": b, "bank_index": j,
                "ref_p": scores[j]}
        replay_cases.append({"case_id": case_id(cid, arm_name, s, j), "kind": "replay", **spec(j),
                             "target_commitment": res["commitment"], "meta": meta})
    (runner.EXP / "banks").mkdir(exist_ok=True)
    runner.write_json(runner.EXP / "banks" / f"{cid}.{arm_name}.json", bank)
    with open(runner.EXP / "banks" / f"{cid}.{arm_name}.replays.jsonl", "w") as f:
        for c in replay_cases:
            f.write(json.dumps(c) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: v for k, v in bank.items() if k not in ("scores", "draws", "bins")}))


if __name__ == "__main__":
    main()
