"""Extend one original's raw subsample bank and draw additional replays (29 October 2026 extension).

Loads banks/<cid>.subsample-raw.json (the frozen stage-1/2 bank), keeps every score in it,
scores further subsamples j = bank_size, bank_size+1, ... with the original's frozen
score.py (same seed recipe, so index j always denotes the same subsample), growing in
chunks of --chunk until the tail holds --tail-extra members that were not drawn before or
the bank reaches --max-bank. Then draws --tail-extra tail and --body-extra body replays
uniformly without replacement among members not drawn in the base design, exactly the
raw estimator's draw. Nothing in the base bank or its replay cases is modified.

Outputs: banks/<cid>.subsample-raw-ext1.json and banks/<cid>.subsample-raw-ext1.replays.jsonl
(arm label "subsample-raw-ext1", meta.budget "ext1").

Usage (one original per array task):
  python extend_references.py --task N --originals-list main_originals_pooled99.jsonl --n-select 99
      [--tail-extra 200] [--body-extra 200] [--chunk 2000] [--max-bank 128000] [--concurrency 32]
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import random
import shutil
from pathlib import Path

import make_references as m
import runner

BASE_ARM = "subsample-raw"
ARM = "subsample-raw-ext1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--originals-list", default="main_originals_pooled99.jsonl")
    ap.add_argument("--n-select", type=int, default=99)
    ap.add_argument("--tail-extra", type=int, default=200)
    ap.add_argument("--body-extra", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-bank", type=int, default=128_000)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()

    chosen = m.selected_originals(a.originals_list, a.n_select)
    if a.task >= len(chosen):
        return
    case, res = chosen[a.task]
    cid, slot, seed, u = case["case_id"], case["meta"]["slot"], case["meta"]["seed"], res["u"]
    base = json.loads((runner.EXP / "banks" / f"{cid}.{BASE_ARM}.json").read_text())
    assert base["original"] == cid and base["seed"] == seed and base["u"] == u, "base bank mismatch"
    mm, n = base["m"], base["n"]
    out_json = runner.EXP / "banks" / f"{cid}.{ARM}.json"
    if out_json.exists():
        print("already extended", cid, slot)
        return
    workspace = runner.EXP / "originals" / cid / "workspace"
    header, rows = m.read_rows(case["data_path"])
    assert len(rows) == n
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"bank-{ARM}-{cid}"
    tmp.mkdir(parents=True, exist_ok=True)

    scores = {int(j): p for j, p in base["scores"].items()}
    used = {d["bank_index"] for d in base["draws"]}
    J0 = base["bank_size"]
    assert max(scores) == J0 - 1 and len(scores) == J0, "base bank scores incomplete"

    def score(j):
        path = tmp / f"ref-{j}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows[i] for i in runner.subsample_indices(seed, j, n, mm))
        r = runner.score_dataset(workspace, str(path), tmp / f"run-{j}")
        shutil.rmtree(tmp / f"run-{j}", ignore_errors=True)
        path.unlink()
        return j, runner.p_from(r["result"])

    def strata():
        tail = sorted(j for j, p in scores.items() if p is not None and p <= u and j not in used)
        body = sorted(j for j, p in scores.items() if p is not None and p > u and j not in used)
        return tail, body

    J = J0
    tail, body = strata()
    while (len(tail) < a.tail_extra or len(body) < a.body_extra) and J < a.max_bank:
        J_new = min(a.max_bank, J + a.chunk)
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            scores.update(pool.map(score, range(J, J_new)))
        J = J_new
        tail, body = strata()
        print(json.dumps({"slot": slot, "bank_size": J, "tail_available": len(tail), "body_available": len(body)}), flush=True)

    pick = random.Random(f"{seed}-{ARM}-select")
    pick.shuffle(tail)
    pick.shuffle(body)
    draws = [("tail", m.bin_of(scores[j], 0.0, u, base["nb1"]), j) for j in tail[:a.tail_extra]] + \
            [("body", m.bin_of(scores[j], u, 1.0, base["nb0"]), j) for j in body[:a.body_extra]]
    valid = [p for p in scores.values() if p is not None]
    n_tail = sum(1 for p in valid if p <= u)
    bank = {"original": cid, "slot": slot, "arm": ARM, "draw": "raw", "u": u, "seed": seed, "m": mm, "n": n,
            "base_arm": BASE_ARM, "base_bank_size": J0, "base_draws": len(used),
            "tail_extra_requested": a.tail_extra, "body_extra_requested": a.body_extra,
            "bank_size": J, "n_scored": len(valid), "n_tail": n_tail, "n_body": len(valid) - n_tail,
            "tail_drawn": sum(1 for s, _, _ in draws if s == "tail"),
            "body_drawn": sum(1 for s, _, _ in draws if s == "body"),
            "tail_shortfall": max(0, a.tail_extra - len(tail)), "body_shortfall": max(0, a.body_extra - len(body)),
            "bank_tail_frac_over_u": (n_tail / max(1, len(valid))) / u if u > 0 else None,
            "max_bank": a.max_bank, "chunk": a.chunk, "nb0": base["nb0"], "nb1": base["nb1"]}
    bank["scores"] = {str(j): p for j, p in sorted(scores.items())}
    bank["draws"] = [{"stratum": s, "bin": b, "bank_index": j} for s, b, j in draws]
    replay_cases = []
    for s, b, j in draws:
        meta = {"original": cid, "slot": slot, "arm": ARM, "budget": "ext1", "stratum": s, "bin": b,
                "bank_index": j, "ref_p": scores[j]}
        spec = {"subsample": {"source": case["data_path"], "seed": seed, "bank_index": j, "m": mm}}
        replay_cases.append({"case_id": m.case_id(cid, ARM, s, j), "kind": "replay", **spec,
                             "target_commitment": res["commitment"], "meta": meta})
    runner.write_json(out_json, bank)
    with open(runner.EXP / "banks" / f"{cid}.{ARM}.replays.jsonl", "w") as f:
        for c in replay_cases:
            f.write(json.dumps(c) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: v for k, v in bank.items() if k not in ("scores", "draws")}))


if __name__ == "__main__":
    main()
