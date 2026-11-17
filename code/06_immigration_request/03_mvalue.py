#!/usr/bin/env python3
"""Miao et al.'s m-value of the realistic request's original z (t0) among the analyses logged by 60 full-data
reference sessions, with each session's lead and the clustering its plans state.

Inputs:  data/agent_runs/issp_mvalue_luna/sessions.csv, analysis_paths.jsonl.gz,
         data/intermediate/06_immigration_request/original_full_data.json
Output:  data/derived/06_immigration_request/mvalue.json
"""
import collections
import csv
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "data/agent_runs/issp_mvalue_luna"
OUT = ROOT / "data/derived/06_immigration_request"
BRIEFS = ("pro_team", "anti_team", "nonpartisan")
TIE = 1e-8  # a lead with |z - t0| at or below this is the original's regression, re-estimated

EST = re.compile(r"^(estimates?|numerical_estimates?|point_estimate|coef|coefficients?|beta|slope|ame)(?=_|$)", re.I)
SE = re.compile(r"^(std_error|standard_error|se|stderr|std_err|robust_se|cluster_se)(?=_|$)", re.I)
RAW = re.compile(r"1pp|per_pp|1_percentage|raw|^$")
# nested keys containing any of these substrings are not searched ("ci" also skips "specification" and "decision")
SKIP = re.compile(r"diagnostic|sample|ci|confidence|interval|interpret|range|min|max|mean|comparison", re.I)
COUNTRY_WAVE = re.compile(r"cntry_wave|country[- ]wave|country[- ]year")


def num(x):
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def level(d):
    """At most one analysis from the scalar fields of one object. A field named "value" is the estimate when a
    standard error sits beside it."""
    est = {EST.sub("", k): num(v) for k, v in d.items() if EST.match(k) and num(v) is not None}
    if not est and num(d.get("value")) is not None and any(SE.match(k) for k in d):
        est = {"": num(d["value"])}
    se = {SE.sub("", k): num(v) for k, v in d.items() if SE.match(k) and num(v) is not None and num(v) > 0}
    if not est:
        return None
    key = next((s for s in est if s in se), None)
    if key is None:
        key = next((s for s in est if RAW.search(s)), next(iter(est)))
    s = se.get(key) if key in se else (next(iter(se.values())) if len(se) == 1 else None)
    if s is None:
        return None
    return {"estimate": est[key], "se": s, "z": est[key] / s}


def walk(obj, out):
    if isinstance(obj, dict):
        got = level(obj)
        if got:
            out.append(got)
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and not SKIP.search(k):
                walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, out)


def clustering(plan):
    """Planned inference of one plan event, from its uncertainty fields."""
    low = " ".join(str(v) for k, v in plan.items()
                   if any(t in k.lower() for t in ("uncert", "infer", "standard_error", "variance", "cluster"))).lower()
    if "cluster" not in low:
        return "other_or_unstated"
    rest = COUNTRY_WAVE.sub(" ", low)  # "country-wave" must not count as "country"
    if "country" in rest or "iso" in rest:
        return "country_clustered"
    return "country_wave_clustered" if COUNTRY_WAVE.search(low) else "clustered_other"


def analyses(text):
    """(analyses, final path id, planned inference classes) of one session's analysis log. An analysis has in_pool
    False when a correction names its result without replacing it."""
    events, plans = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            events.append(e)
    by_id = {e.get("event_id"): e for e in events}
    replaced, unreplaced = {}, set()
    for c in (e for e in events if e.get("event_type") == "correction"):
        target = by_id.get(c.get("original_event_id"))
        if not target or target.get("event_type") != "result":
            continue
        rec = c.get("corrected_record")
        if isinstance(rec, dict) and rec.get("event_type") == "result":
            replaced[target["event_id"]] = rec
        else:
            unreplaced.add(target["event_id"])
    plan_class = {}
    for e in events:
        if e.get("event_type") == "plan":
            plans.append(clustering(e))
            plan_class[e.get("path_id")] = plans[-1]
    rows = []
    for e in events:
        if e.get("event_type") != "result":
            continue
        in_pool = e.get("event_id") not in unreplaced
        e = replaced.get(e.get("event_id"), e)
        if e.get("status") != "estimated":
            continue
        found = []
        walk({k: v for k, v in e.items() if k not in ("event_id", "event_type", "path_id", "status")}, found)
        rows += [{"path_id": e.get("path_id"), "plan_inference": plan_class.get(e.get("path_id"), "no_plan"),
                  "in_pool": in_pool, **f} for f in found]
    finals = [e for e in events if e.get("event_type") == "final_choice"]
    final_path = finals[-1].get("path_id") or finals[-1].get("selected_path_id") if finals else None
    return rows, final_path, plans


def m_value(z, t, w=None):
    z = np.asarray(z, float)
    w = np.ones_like(z) if w is None else np.asarray(w, float)
    return float(min(1.0, 2 * min(w[z <= t].sum(), w[z >= t].sum()) / w.sum()))


def main():
    o = json.loads((ROOT / "data/intermediate/06_immigration_request/original_full_data.json").read_text(
        encoding="utf-8"))
    t0 = o["raw_coefficient_per_1pp_stock"] / o["raw_std_error_per_1pp_stock"]
    briefs = {r["session_id"]: r["brief"] for r in csv.DictReader(open(REF / "sessions.csv", encoding="utf-8"))}
    with gzip.open(REF / "analysis_paths.jsonl.gz", "rt", encoding="utf-8") as fh:
        logs = {r["session_id"]: r for r in map(json.loads, fh)}
    sessions, plans = [], collections.Counter()
    for sid, brief in briefs.items():
        rows, final_path, p = analyses(logs[sid]["analysis_paths"])
        plans.update(p)
        try:
            final = json.loads(logs[sid]["results_final"] or "{}")
        except ValueError:
            final = {}
        final = final if isinstance(final, dict) else {}
        reported = []
        walk(final, reported)
        sessions.append({"brief": brief, "rows": rows,
                         "lead_path": final_path or final.get("selected_path_id") or final.get("path_id"),
                         "reported_z": reported[0]["z"] if reported else None})
    leads = collections.Counter()  # sessions per brief whose lead is the original's regression
    for s in sessions:
        cand = [r for r in s["rows"] if r["path_id"] == s["lead_path"]]
        if cand:
            lead = cand[0] if s["reported_z"] is None else min(cand, key=lambda r: abs(r["z"] - s["reported_z"]))
            leads[s["brief"]] += abs(lead["z"] - t0) <= TIE
    pools = [[r["z"] for r in s["rows"] if r["in_pool"]] for s in sessions]
    z = [v for pool in pools for v in pool]
    w = [1 / len(pool) for pool in pools for _ in pool]
    result = {
        "t0": t0, "sessions": len(sessions),
        "sessions_by_brief": {b: sum(s["brief"] == b for s in sessions) for b in BRIEFS},
        "analyses": len(z), "share_at_or_below_t0": float(np.mean(np.array(z) <= t0)),
        "m": m_value(z, t0), "m_sessions_weighted_equally": m_value(z, t0, w),
        "leads_original_by_brief": {b: int(leads[b]) for b in BRIEFS},
        "planned_inference": dict(sorted(plans.items())),
        "analyses_by_plan_inference": dict(sorted(collections.Counter(
            r["plan_inference"] for s in sessions for r in s["rows"] if r["in_pool"]).items()))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mvalue.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
