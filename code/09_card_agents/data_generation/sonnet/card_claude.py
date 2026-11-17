#!/usr/bin/env python3
"""Card cohort on Claude Sonnet 5.5 (../MAIN_PLAN.md) on the authors' two Claude subscriptions.

The guest, the Claude CLI call, the secret handling and the record sink are the multi-model originals'
(common.py and run_claude.py, copied unchanged from output/multi-model-originals-20261030/code). This file
adds the Card packet (data.csv, CODEBOOK.md, DESIGN.md), the three Card prompts, the R library the Card
cohorts used, a stop at 95% of an account's weekly meter, and the Sol cohort's failure rule: no retries,
except a turn refused at a usage or rate limit before the agent acted; a session cut by a meter stop or
a killed job counts as not done and runs again in a later launch.

  python3 card_claude.py original --account a
  python3 card_claude.py split --accounts a,b --name L1
  python3 card_claude.py replays --account a --pending pending-L1-a.json --workers 10   # any number of tasks
  python3 card_claude.py status
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MMO_ROOT", str(HERE.parent))
import common  # noqa: E402
import run_claude  # noqa: E402

RUN = common.RUN
INPUTS = RUN / "inputs"
RLIB = Path(os.environ.get("CARD_RLIB", "<cluster-project>/terra-rlib-4.5.1"))
MODEL = "claude-sonnet-5-5"
PAUSE_AT = 0.95  # weekly utilization at which an account stops claiming new cases
COMPARISON_SCHEMA = {"type": "object", "properties": {"model_null_match": {"type": "boolean"}, "reason": {"type": "string"}},
                     "required": ["model_null_match", "reason"], "additionalProperties": False}
RETRY = ("unstarted_limit",)                       # requeued within a launch
RERUN = ("unstarted_limit", "interrupted_limit")   # not done: a later launch runs them again
STOP = RUN / "STOP"
LOCK = threading.Lock()
common.TIMEOUT["comparison"] = common.TIMEOUT["commitment"]
run_claude.RATE_STOP = ("rejected",)  # the meter reading below, not the CLI's early warning, pauses an account


class CardRoute(run_claude.ClaudeRoute):
    def __init__(self, account):
        self.model, self.account, self.paused, self.utilization = MODEL, account, False, None
        self.token = (RUN / "private" / f"claude-token-{account}").read_text().strip()
        self.secret_env = {"CLAUDE_CODE_OAUTH_TOKEN": self.token}
        self.label = f"claude_subscription:{account}"

    def guest_env(self):
        return dict(super().guest_env(), R_LIBS="/home/researcher/Rlib:/opt/rlib")

    def ro_binds(self):
        return dict(super().ro_binds(), **{"/opt/rlib": RLIB})

    def dirs(self):
        return super().dirs() + ("/opt/rlib",)

    def prepare(self, c):
        super().prepare(c)
        (c / "home" / "Rlib").mkdir(parents=True, exist_ok=True)

    def phase(self, c, phase, prompt, session_id=None, schema=None):
        rec = super().phase(c, phase, prompt, session_id, schema)
        for e in rec.get("rate_events") or []:
            u = (((e or {}).get("unifiedWindows") or {}).get("seven_day") or {}).get("utilization")
            if u is not None:
                self.utilization = u
                if u >= PAUSE_AT:
                    self.paused = True
        return rec


def classify(rec, replay):
    if rec.get("runner_error"):
        return "unavailable", "runner"
    for name in ("analysis", "commitment") + (("comparison",) if replay else ()):
        ph = rec["phases"].get(name)
        if ph is None:
            return "unavailable", f"no_{name}"
        if not ph.get("ok"):
            kind = ph.get("fault_kind") or "cut_off"
            if kind in ("usage_limit", "rate_limit"):
                unstarted = name == "analysis" and not (ph.get("total_cost_usd") or 0) > 0
                return ("unstarted_limit" if unstarted else "interrupted_limit"), kind
            return "unavailable", kind
    for key in ("commitment_error", "verdict_error"):
        if rec.get(key):
            return "unavailable", key
    if rec.get("session_models") and rec["session_models"] != [MODEL]:
        return "unavailable", "wrong_model"
    return "complete", None


def run_case(route, case, target, attempt, campaign, work_root):
    """One case: analysis, commitment and, for a replay, the comparison, in one resumed session."""
    cid = case["case_id"]
    c = Path(work_root) / f"{cid}-a{attempt}"
    for name in list(common.GUEST) + ["packet", "logs"]:
        (c / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(case["packet"], c / "packet" / "data.csv")
    for f in ("CODEBOOK.md", "DESIGN.md"):
        shutil.copy2(INPUTS / f, c / "packet" / f)
    route.prepare(c)
    rec = {k: case.get(k) for k in ("case_id", "position", "bank_id", "stratum", "bank_p")}
    rec.update(model=MODEL, route=route.label, attempt=attempt, effort=common.EFFORT, started=time.time(),
               host=socket.gethostname(), data_sha256=common.sha(c / "packet" / "data.csv"), phases={})
    try:
        a = route.phase(c, "analysis", (INPUTS / "research-prompt.txt").read_text())
        rec["phases"]["analysis"] = a
        if a.get("ok"):
            rec["final_report"] = a["answer"][:20000]
            cm = route.phase(c, "commitment", (INPUTS / "commitment_prompt.txt").read_text(),
                             session_id=a["session_id"], schema=common.COMMITMENT_SCHEMA)
            rec["phases"]["commitment"] = cm
            if cm.get("ok"):
                try:
                    rec["commitment"] = common.parse_commitment(cm["answer"])
                except (ValueError, TypeError) as e:
                    rec["commitment_error"] = str(e)
                if target is not None and "commitment" in rec:
                    prompt = (INPUTS / "comparison_prompt.txt").read_text().replace(
                        "{{TARGET_JSON}}", json.dumps(target, ensure_ascii=False, separators=(",", ":")))
                    cp = route.phase(c, "comparison", prompt, session_id=a["session_id"], schema=COMPARISON_SCHEMA)
                    rec["phases"]["comparison"] = cp
                    if cp.get("ok"):
                        try:
                            v = json.loads(cp["answer"])
                            if set(v) != {"model_null_match", "reason"} or type(v["model_null_match"]) is not bool:
                                raise ValueError("malformed comparison")
                            rec["verdict"] = v
                        except (ValueError, TypeError) as e:
                            rec["verdict_error"] = str(e)
        rec["has_analysis_r"] = (c / "workspace" / "analysis.R").exists()
    except Exception as e:  # logged as a fault
        rec["runner_error"] = f"{type(e).__name__}: {e}"
    finally:
        rec["finished"] = time.time()
    try:
        rec.update(route.facts(c))
    except Exception as e:
        rec["facts_error"] = f"{type(e).__name__}: {e}"
    rec["data_unchanged"] = common.sha(c / "packet" / "data.csv") == rec["data_sha256"]
    rec["meter"] = route.utilization
    rec["status"], rec["fault_kind"] = classify(rec, target is not None)
    if target is None and rec["status"] == "complete":
        keep_original(c, rec)
    rec["token_scrubbed_files"] = common.scrub_case(c, route.secrets())
    common.sink(rec, c, route, campaign)
    shutil.rmtree(c, ignore_errors=True)
    return rec


def keep_original(c, rec):
    out = RUN / "original"
    if (out / "result.json").exists():
        raise SystemExit("an original is already committed")
    shutil.copytree(c / "workspace", out / "workspace", symlinks=True, ignore=shutil.ignore_patterns("data.csv"))
    if rec.get("has_analysis_r"):
        shutil.copy2(c / "workspace" / "analysis.R", out / "emitted-analysis.R")
    common.write_json(out / "result.json", {"status": "committed", "model": MODEL, "effort": common.EFFORT,
                                           "case_id": rec["case_id"], "attempt": rec["attempt"],
                                           "commitment": rec.get("commitment"), "has_analysis_r": rec.get("has_analysis_r"),
                                           "final_report": rec.get("final_report"), "route": rec["route"]})


def kept():
    """The first final record per case; a later record replaces only one that was not done."""
    out = {}
    for p in sorted((RUN / "node-results").glob("*.results.jsonl")):
        for r in common.read_jsonl(p):
            prev = out.get(r["case_id"])
            if prev is None or prev["status"] in RERUN:
                out[r["case_id"]] = r
    return out


def ensure_dirs():
    for d in ("node-results", "node-archives", "node-status"):
        (RUN / d).mkdir(exist_ok=True)


def claim(claims, case):
    """One claim file per case and launch, so any number of tasks can share one pending list."""
    if claims is None:
        return True
    try:
        fd = os.open(claims / case["case_id"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')} {time.time()}\n".encode())
    os.close(fd)
    return True


def launch(route, cases, target, workers, campaign, claims=None):
    ensure_dirs()
    if claims is not None:
        claims.mkdir(parents=True, exist_ok=True)
    work_root = Path(os.environ.get("TMPDIR", "/tmp")) / campaign
    work_root.mkdir(parents=True, exist_ok=True)
    queue = collections.deque((c, 1) for c in cases)
    done = collections.Counter()

    def status():
        common.write_json(RUN / "node-status" / f"{campaign}.json",
                          {"campaign": campaign, "assigned": len(cases), "done": dict(done), "queue": len(queue),
                           "paused": route.paused, "meter": route.utilization, "time": time.time(), "host": socket.gethostname()})

    def worker(i):
        time.sleep(i * route.start_gap)
        while not (STOP.exists() or route.paused):
            with LOCK:
                item = queue.popleft() if queue else None
            if item is None:
                return
            case, attempt = item
            if attempt == 1 and not claim(claims, case):
                continue
            try:
                rec = run_case(route, case, target, attempt, campaign, work_root)
            except Exception as e:  # no record: not done, a later launch runs it again
                print(f"case {case['case_id']} attempt {attempt} failed in the runner: {e!r}", flush=True)
                done["runner_exception"] += 1
                continue
            done[rec["status"]] += 1
            status()
            if rec["status"] in RETRY and attempt < 5 and not route.paused:
                time.sleep(60)
                with LOCK:
                    queue.append((case, attempt + 1))

    status()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(worker, i) for i in range(workers)]:
            f.result()
    status()
    print(json.dumps({"campaign": campaign, "assigned": len(cases), "done": done, "left": len(queue),
                      "paused": route.paused, "meter": route.utilization, "stopped": STOP.exists()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("original", "split", "replays", "status"))
    ap.add_argument("--account", default="a")
    ap.add_argument("--accounts", help="split: the accounts sharing the launch, in roster order")
    ap.add_argument("--name", help="split: the launch name")
    ap.add_argument("--pending", help="replays: this account's pending file from split")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, help="checks only: the first N pending cases")
    a = ap.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "local")
    if a.mode == "status":
        k = kept()
        roster = json.loads((RUN / "roster.json").read_text())["cases"] if (RUN / "roster.json").exists() else []
        print(json.dumps({"records": len(k), "statuses": collections.Counter(r["status"] for r in k.values()),
                          "roster": len(roster), "pending": sum(1 for c in roster if k.get(c["case_id"], {}).get("status") in (None,) + RERUN)}))
        return
    route = CardRoute(a.account) if a.mode != "split" else None
    if a.mode == "original":
        if (RUN / "original" / "result.json").exists():
            raise SystemExit("an original is already committed")
        ensure_dirs()
        case = {"case_id": "original", "position": 0, "packet": str(INPUTS / "data.csv")}
        for attempt in range(1, 4):
            rec = run_case(route, case, None, attempt, f"original-{a.account}-{job}", Path(os.environ.get("TMPDIR", "/tmp")) / f"original-{job}")
            print(json.dumps({"attempt": attempt, "status": rec["status"], "fault": rec["fault_kind"], "meter": route.utilization}), flush=True)
            if rec["status"] == "complete" or route.paused:
                break
        return
    if a.mode == "split":
        # Written once per launch, so arrays that start at different times share one split.
        roster = sorted(json.loads((RUN / "roster.json").read_text())["cases"], key=lambda c: c["position"])
        k = kept()
        pending = [c for c in roster if k.get(c["case_id"], {}).get("status") in (None,) + RERUN]
        accounts = a.accounts.split(",")
        for i, name in enumerate(accounts):
            path = RUN / f"pending-{a.name}-{name}.json"
            with open(path, "x") as f:
                json.dump({"launch": a.name, "account": name, "accounts": accounts, "cases": pending[i::len(accounts)]}, f, indent=1)
            print(path.name, len(pending[i::len(accounts)]))
        return
    target = json.loads((RUN / "original" / "result.json").read_text())["commitment"]
    k = kept()
    pending = json.loads((RUN / a.pending).read_text())
    cases = [c for c in pending["cases"] if k.get(c["case_id"], {}).get("status") in (None,) + RERUN]
    if a.limit:
        cases = cases[:a.limit]
    # Every task walks the whole list in roster order and runs only the cases it claims first.
    launch(route, cases, target, a.workers, f"replays-{a.account}-task{a.task}-{job}",
           claims=RUN / "claims" / f"{pending['launch']}-{a.account}")


if __name__ == "__main__":
    main()
