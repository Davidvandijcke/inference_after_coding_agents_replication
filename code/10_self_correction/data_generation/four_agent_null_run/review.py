#!/usr/bin/env python3
"""Blind review of the report openings: one fresh session per read, on a ChatGPT Pro login.

The session has no shell and no files. The prompt is the whole packet and the answer comes back
through the output schema. A refused or cut-off turn is retried, as in the runners. The reviewer is
the authors' choice: gpt-6.1-sol for every read, or gpt-6.1-terra for the first reads with gpt-6.1-sol
for the 40 repeats of each task as a cross-check. --read 1 or 2 selects the reads a launch takes.

  python3 review.py --model gpt-6.1-terra --account second --read 1 --task 0 --tasks 2 --workers 10
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

import common
import run_codex

FRAMING = {"type": "object", "properties": {"category": {"type": "string", "enum": ["STATES", "HEDGES", "DISCLAIMS"]},
                                            "quote": {"type": "string"}},
           "required": ["category", "quote"], "additionalProperties": False}
AGREEMENT = {"type": "object", "properties": {"agrees": {"type": "string", "enum": ["YES", "NO", "UNCLEAR"]},
                                              "differs_on": {"type": "array", "items": {"type": "string", "enum": ["outcome", "followup", "population", "contrast"]}},
                                              "quote": {"type": "string"}},
             "required": ["agrees", "differs_on", "quote"], "additionalProperties": False}
STANCE = {"type": "object", "properties": {"category": {"type": "string", "enum": ["STANDS_BEHIND", "WARNS", "DISCLAIMS"]},
                                           "quote": {"type": "string"}},
          "required": ["category", "quote"], "additionalProperties": False}
FACTS = {"type": "object",
         "properties": {"tests_stated": {"type": ["integer", "null"]}, "other_significant": {"type": "string", "enum": ["YES", "NO"]},
                        "own_correction": {"type": "string", "enum": ["NONE", "BONFERRONI", "HOLM", "FDR", "PERMUTATION", "OTHER"]},
                        "correction_tests": {"type": ["integer", "null"]}, "adjusted_p": {"type": ["number", "null"]},
                        "survives": {"type": "string", "enum": ["YES", "NO", "NA"]}, "quote": {"type": "string"}},
         "required": ["tests_stated", "other_significant", "own_correction", "correction_tests", "adjusted_p", "survives", "quote"],
         "additionalProperties": False}
DISCLOSURE = {"type": "object", "properties": {"disclosure": {"type": "string", "enum": ["NUMBER", "WORDS", "NONE"]},
                                               "warning": {"type": "string", "enum": ["YES", "NO"]}, "quote": {"type": "string"}},
              "required": ["disclosure", "warning", "quote"], "additionalProperties": False}
FACTS_CARD = {"type": "object", "properties": {"nominal_p": {"type": ["number", "null"]}, **FACTS["properties"]},
              "required": ["nominal_p"] + FACTS["required"], "additionalProperties": False}
SCHEMA = {"framing": FRAMING, "agreement": AGREEMENT, "opening": STANCE, "report": STANCE, "facts": FACTS, "disclosure": DISCLOSURE,
          "facts_card": FACTS_CARD}
REVIEW = common.RUN / os.environ.get("MMO_REVIEW", "review")
OUT = REVIEW / "out"


class ReviewRoute(run_codex.CodexRoute):
    def config_text(self):
        text = super().config_text()
        for key in ("features.shell_tool", "features.unified_exec", "features.code_mode_host"):
            text = text.replace(f"{key} = true", f"{key} = false")
        return text


def read_one(case, route, work, results):
    if (REVIEW / "STOP").exists() or route.paused:
        return
    key = f"{case['id']}-{case['read']}"
    for attempt in range(1, common.MAX_ATTEMPTS + 1):
        c = Path(work) / f"{key}-a{attempt}"
        for name in list(common.GUEST) + ["logs"]:
            (c / name).mkdir(parents=True, exist_ok=True)
        route.prepare(c)
        rec = {"id": case["id"], "task": case["task"], "read": case["read"], "position": case["position"], "attempt": attempt,
               "reviewer": route.model, "started": time.time()}
        try:
            r = route.phase(c, "analysis", case["prompt"], schema=SCHEMA[case["task"]])
            rec.update({k: r.get(k) for k in ("ok", "fault_kind", "error", "usage", "seconds")})
            if r.get("ok"):
                rec["label"] = common.json.loads(r["answer"])
            else:
                logs = c / "logs"
                rec["stderr_tail"] = (logs / "analysis.stderr").read_text(errors="replace")[-600:]
                rec["events_tail"] = (logs / "analysis.events.jsonl").read_text(errors="replace")[-600:]
        except Exception as e:
            rec.update(ok=False, fault_kind="runner", error=f"{type(e).__name__}: {e}")
        finally:
            route.cleanup(c)
        rec.update(route.facts(c))
        line = json.dumps(rec)
        for s in route.secrets():
            line = line.replace(s, "<redacted>")
        with common.SINK_LOCK, open(results, "a") as f:
            f.write(line + "\n")
        shutil.rmtree(c, ignore_errors=True)
        if rec.get("label"):
            return
        if route.paused:
            return
        time.sleep(20 * attempt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("gpt-6.1-sol", "gpt-6.1-terra"))
    ap.add_argument("--account", required=True, choices=("primary", "second"))
    ap.add_argument("--read", type=int, choices=(1, 2), help="only first reads or only repeats; default both")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    route = ReviewRoute(a.model, a.account)
    OUT.mkdir(parents=True, exist_ok=True)
    cases = common.read_jsonl(REVIEW / "cases.jsonl")
    done = set()
    for p in OUT.glob("results-*.jsonl"):
        done |= {f"{r['id']}-{r['read']}" for r in common.read_jsonl(p) if r.get("label")}
    cases = [c for c in cases[:a.limit] if f"{c['id']}-{c['read']}" not in done and (a.read is None or c["read"] == a.read)][a.task::a.tasks]
    work = Path(os.environ.get("TMPDIR", "/tmp")) / f"review-{os.environ.get('SLURM_JOB_ID', 'local')}-{a.task}"
    work.mkdir(parents=True, exist_ok=True)
    results = OUT / f"results-{os.environ.get('SLURM_JOB_ID', 'local')}-{a.task}.jsonl"
    time.sleep(10 * a.task)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(lambda c: read_one(c, route, work, results), cases))
    print(json.dumps({"task": a.task, "assigned": len(cases), "paused": route.paused}), flush=True)
