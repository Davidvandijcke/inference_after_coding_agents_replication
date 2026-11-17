"""Score a subsample bank in small array tasks and let the frozen builder reuse the scores.

Scheduling only (the authors, 30 October: small tasks backfill, near-whole-node requests wait). A
chunk scores subsamples lo to hi - 1 of one original exactly as make_references.score does:
same indices, same file, same sandbox. The build step then runs make_references.main()
unchanged, with runner.score_dataset answering from the saved chunks, so the doubling rule,
the seeded draws and the output files are the frozen ones. A j with no saved score falls
through to the real sandbox.

  plan    write the next chunks for the originals whose tail or body is not yet full
  score   run one chunk (array task); --fast loads the libraries once per sandbox (fast_guest.py)
  verify  compare the fast path with the saved standard scores of one original
  settle  verify, score until the doubling rule stops, then the program's fresh and null checks
  build   run the frozen builder on the saved scores

Usage (EXP_ROOT set, from the runtime folder):
  python bank_chunks.py plan --originals-list L --out PLAN.jsonl [--chunk 2000] [--first 4000]
  python bank_chunks.py score --plan PLAN.jsonl --task K --concurrency 4 [--fast]
  python bank_chunks.py verify --originals-list L --task N --count 400 --concurrency 4
  python bank_chunks.py build --originals-list L --tail 40 --body 200 --m 200 --max-bank 128000
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import generate
import make_references as mr
import runner

CHUNKS = runner.EXP / "bank-chunks"


def saved_scores(cid):
    out = {}
    for p in CHUNKS.glob(f"{cid}.*.json"):
        out.update({int(j): v for j, v in json.loads(p.read_text())["scores"].items()})
    return out


def next_target(scores, u, tail, body, cap):
    """The next bank size to score up to, or None when the doubling rule already stops."""
    J = 2000
    while True:
        have = [p for j, p in scores.items() if j < J and p is not None]
        n_tail = sum(p <= u for p in have)
        if J <= (max(scores) + 1 if scores else 0) and ((n_tail >= tail and len(have) - n_tail >= body) or J >= cap):
            return None
        if J > (max(scores) + 1 if scores else 0):
            break
        J *= 2
    n = len([p for p in scores.values() if p is not None])
    n_tail = sum(p <= u for p in scores.values() if p is not None)
    if n == 0:
        return J
    if n_tail == 0:
        return cap  # nothing yet in the tail: only the cap can settle it
    while J < cap and n_tail / n * J < 1.25 * tail:
        J *= 2
    return min(J, cap)


def cmd_plan(a):
    plan, status = [], []
    for case, res in mr.selected_originals(a.originals_list, 200):
        cid, u = case["case_id"], res["u"]
        scores = saved_scores(cid)
        target = a.first if not scores else next_target(scores, u, a.tail, a.body, a.max_bank)
        done = max(scores) + 1 if scores else 0
        status.append({"slot": case["meta"]["slot"], "scored": done, "target": target})
        if target:
            for lo in range(done, target, a.chunk):
                plan.append({"case_id": cid, "seed": case["meta"]["seed"], "data_path": case["data_path"], "lo": lo,
                             "hi": min(target, lo + a.chunk), "m": a.m})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in plan)
    print(json.dumps({"chunks": len(plan), "subsamples": sum(x["hi"] - x["lo"] for x in plan),
                      "unsettled": [s for s in status if s["target"]], "settled": sum(s["target"] is None for s in status)}))


def write_ref(path, header, rows, seed, j, m):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows[i] for i in runner.subsample_indices(seed, j, len(rows), m))


def fast_batch(workspace, writer, js, run):
    """Score the datasets js in one sandbox with fast_guest.py; the workspace is copied as in score_dataset."""
    for name in runner.GUEST:
        (run / name).mkdir(parents=True, exist_ok=True)
    for f in Path(workspace).iterdir():
        if f.name == "data.csv":
            continue
        if f.is_file() and f.stat().st_size < 20_000_000:
            shutil.copy2(f, run / "workspace" / f.name)
        elif f.is_dir() and f.name not in ("__pycache__",):
            shutil.copytree(f, run / "workspace" / f.name, symlinks=True,
                            ignore=shutil.ignore_patterns("*.csv", "*.parquet", "*.rds", "*.RData"))
    data, out = run / "scoredata", run / "scoreout"
    data.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    for j in js:
        writer(data / f"ref-{j}.csv", j)
    guest = Path(__file__).resolve().parent / "fast_guest.py"
    cmd = runner.bwrap(run, ["python3", "/opt/fast/fast_guest.py", "/scoredata", "/scoreout"], network=False,
                       extra_ro={"/opt/fast/fast_guest.py": guest, "/scoredata": data}, extra_rw={"/scoreout": out})
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * 3600)
    if not (out / "results.json").exists():
        raise RuntimeError(f"fast scorer failed: {p.stderr[-800:]}")
    results = json.loads((out / "results.json").read_text())
    shutil.rmtree(run, ignore_errors=True)
    return {int(j): runner.p_from(r) for j, r in results.items()}


def fast_scores(workspace, writer, js, tmp, concurrency):
    parts = [js[k::concurrency] for k in range(concurrency) if js[k::concurrency]]
    with cf.ThreadPoolExecutor(max_workers=len(parts)) as pool:
        done = pool.map(lambda kp: fast_batch(workspace, writer, kp[1], tmp / f"fast-{kp[0]}"), enumerate(parts))
    out = {}
    for d in done:
        out.update(d)
    return out


def cmd_score(a):
    item = [json.loads(l) for l in open(a.plan) if l.strip()][a.task]
    cid, seed, lo, hi = item["case_id"], item["seed"], item["lo"], item["hi"]
    dest = CHUNKS / f"{cid}.{lo}.json"
    if dest.exists():
        return
    workspace = runner.EXP / "originals" / cid / "workspace"
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"chunk-{cid}-{lo}"
    tmp.mkdir(parents=True, exist_ok=True)
    header, rows = mr.read_rows(item["data_path"])

    def score(j):
        path = tmp / f"ref-{j}.csv"
        write_ref(path, header, rows, seed, j, item["m"])
        r = runner.score_dataset(workspace, str(path), tmp / f"run-{j}")
        shutil.rmtree(tmp / f"run-{j}", ignore_errors=True)
        path.unlink()
        return j, runner.p_from(r["result"])

    if a.fast:
        scores = fast_scores(workspace, lambda path, j: write_ref(path, header, rows, seed, j, item["m"]),
                             list(range(lo, hi)), tmp, a.concurrency)
    else:
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            scores = dict(pool.map(score, range(lo, hi)))
    shutil.rmtree(tmp, ignore_errors=True)
    runner.write_json(dest, {"case_id": cid, "lo": lo, "hi": hi, "fast": a.fast, "scores": {str(j): p for j, p in sorted(scores.items())}})


def verify_fast(case, count, m, tmp, concurrency):
    """Fast path against the saved standard scores of this original's own program."""
    cid = case["case_id"]
    saved = {}
    for p in CHUNKS.glob(f"{cid}.*.json"):
        rec = json.loads(p.read_text())
        if not rec.get("fast"):
            saved.update({int(j): v for j, v in rec["scores"].items()})
    js = sorted(saved)[:count]
    header, rows = mr.read_rows(case["data_path"])
    writer = lambda path, j: write_ref(path, header, rows, case["meta"]["seed"], j, m)
    fast = fast_scores(runner.EXP / "originals" / cid / "workspace", writer, js, tmp, concurrency)
    same = sum((saved[j] is None and fast[j] is None) or (saved[j] is not None and fast[j] is not None and abs(saved[j] - fast[j]) < 1e-12) for j in js)
    rec = {"slot": case["meta"]["slot"], "case_id": cid, "compared": len(js), "identical": same,
           "max_diff": max((abs(saved[j] - fast[j]) for j in js if saved[j] is not None and fast[j] is not None), default=None),
           "none_standard": sum(saved[j] is None for j in js), "none_fast": sum(fast[j] is None for j in js)}
    runner.write_json(runner.EXP / "bank-verify" / f"{cid}.json", rec)
    return rec


def cmd_verify(a):
    case, _ = mr.selected_originals(a.originals_list, 200)[a.task]
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"verify-{case['case_id']}"
    tmp.mkdir(parents=True, exist_ok=True)
    print(json.dumps(verify_fast(case, a.count, a.m, tmp, a.concurrency)))
    shutil.rmtree(tmp, ignore_errors=True)


def cmd_settle(a):
    """Verify the fast path on this original, then score until the frozen doubling rule stops."""
    case, res = mr.selected_originals(a.originals_list, 200)[a.task]
    cid, seed, u = case["case_id"], case["meta"]["seed"], res["u"]
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"settle-{cid}"
    tmp.mkdir(parents=True, exist_ok=True)
    rec = verify_fast(case, a.count, a.m, tmp, a.concurrency)
    print(json.dumps(rec), flush=True)
    if rec["compared"] == 0 or rec["identical"] != rec["compared"]:
        raise SystemExit("fast path differs from the standard path on this program: not used")
    header, rows = mr.read_rows(case["data_path"])
    workspace = runner.EXP / "originals" / cid / "workspace"
    writer = lambda path, j: write_ref(path, header, rows, seed, j, a.m)
    while True:
        scores = saved_scores(cid)
        target = next_target(scores, u, a.tail, a.body, a.max_bank)
        if target is None:
            break
        for lo in range(max(scores) + 1, target, a.step):
            hi = min(target, lo + a.step)
            got = fast_scores(workspace, writer, list(range(lo, hi)), tmp, a.concurrency)
            runner.write_json(CHUNKS / f"{cid}.{lo}.json", {"case_id": cid, "lo": lo, "hi": hi, "fast": True,
                                                            "scores": {str(j): p for j, p in sorted(got.items())}})
    # The program's power on fresh data under its own effect, and the same seeds with no effect.
    slot = case["meta"]["slot"]
    for name, tau in (("fresh", None), ("null-check", 0.0)):
        dest = runner.EXP / "banks" / f"{cid}.{name}.json"
        if dest.exists():
            continue
        writer_fresh = lambda path, j: generate.write_csv(generate.generate(mr.fresh_seed(slot, j), mr.N_FRESH, tau=tau), path)
        got = fast_scores(workspace, writer_fresh, list(range(a.n_fresh)), tmp, a.concurrency)
        ordered = [got[j] for j in range(a.n_fresh)]
        runner.write_json(dest, {"case_id": cid, "slot": slot, "arm": name, "scores": {str(j): p for j, p in enumerate(ordered)} if name == "fresh" else ordered})
    shutil.rmtree(tmp, ignore_errors=True)


def cmd_build(a):
    real = runner.score_dataset
    for task, (case, _) in enumerate(mr.selected_originals(a.originals_list, 200)):
        if a.task is not None and task != a.task:
            continue
        cache = saved_scores(case["case_id"])

        def from_chunks(workspace, data_path, out_dir):
            j = int(Path(data_path).stem.split("-")[1])
            if j in cache:
                p = cache[j]
                return {"returncode": 0, "result": None if p is None else {"p_value": p}, "stderr": ""}
            return real(workspace, data_path, out_dir)

        runner.score_dataset = from_chunks
        sys.argv = ["make_references.py", "--arm", "subsample", "--draw", "raw", "--task", str(task), "--originals-list",
                    a.originals_list, "--n-select", "200", "--tail", str(a.tail), "--body", str(a.body), "--m", str(a.m),
                    "--max-bank", str(a.max_bank), "--start-bank", str(a.start_bank), "--concurrency", str(a.concurrency)]
        mr.main()
    runner.score_dataset = real


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build", "verify", "settle"):
        s = sub.add_parser(name)
        s.add_argument("--originals-list", default="power_originals.jsonl")
        s.add_argument("--tail", type=int, default=40)
        s.add_argument("--body", type=int, default=200)
        s.add_argument("--m", type=int, default=200)
        s.add_argument("--max-bank", type=int, default=128_000)
        if name == "plan":
            s.add_argument("--out", required=True)
            s.add_argument("--chunk", type=int, default=2000)
            s.add_argument("--first", type=int, default=4000)
        else:
            s.add_argument("--task", type=int)
            s.add_argument("--start-bank", type=int, default=2000)
            s.add_argument("--concurrency", type=int, default=4)
            s.add_argument("--count", type=int, default=2000)
            s.add_argument("--step", type=int, default=16000)
            s.add_argument("--n-fresh", type=int, default=4000)
    s = sub.add_parser("score")
    s.add_argument("--plan", required=True)
    s.add_argument("--task", type=int, required=True)
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    CHUNKS.mkdir(exist_ok=True)
    {"plan": cmd_plan, "score": cmd_score, "build": cmd_build, "verify": cmd_verify, "settle": cmd_settle}[a.cmd](a)
