#!/usr/bin/env python3
"""Collect runtime records and mechanically score the frozen selected event."""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
BANK_CODE = PROJECT / "output/sol-recurrence-power-20261102/bank/build_banks.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


bank = load_module(BANK_CODE, "bank_scoring")
legacy = load_module(HERE / "code_choices.py", "frozen_code_choices")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences(text):
    return re.split(r"(?<=[.!?])\s+", text or "")


def positive_assumption(text, root):
    hits = []
    for s in sentences(text):
        if re.search(root, s, re.I):
            neg = re.search(r"\b(no|not|without|neither|nor|did not|does not|was not|were not|never)\b", s, re.I)
            if not neg:
                hits.append(s)
    return hits


def procedure(text):
    robust = next((x for x in ("HC0", "HC1", "HC2", "HC3") if x in text), None)
    custom_hc2 = bool(re.search(r"1\s*-\s*h|leverage", text, re.I) and re.search(r"t\.(?:sf|cdf)|stats\.t", text))
    if robust is None and custom_hc2: robust = "HC2"
    if re.search(r"use_t\s*=\s*True|t\.(?:sf|cdf)|stats\.t", text): ref = "student_t"
    elif re.search(r"use_t\s*=\s*False|norm\.(?:sf|cdf)|stats\.norm", text): ref = "normal_z"
    else: ref = "unclear"
    return robust, ref


def classify(commitment, score_text):
    model = commitment.get("model") or ""
    null = commitment.get("null") or ""
    outcome = commitment.get("outcome") or ""
    population = commitment.get("population") or ""
    restriction = commitment.get("sample_restrictions") or ""
    coded = legacy.code_one(score_text, commitment)
    out_ok = coded["outcome"] == "employed" and coded["followup"] == "12" and coded["n_outcome_columns"] == 1
    unrestricted = not coded["restricted"]
    # Population wording is a review flag only; actual row restriction is coded
    # from the frozen sample_restrictions field and submitted program.
    pop_ok = bool(re.search(r"participant|adult|trial|evaluation", population, re.I))
    null_ok = bool(re.search(r"tau|beta|treatment|assignment|coefficient", null, re.I) and re.search(r"\b0\b|zero", null, re.I))
    site_ok = bool(re.search(r"site", model, re.I))
    extra_baseline = bool(coded["baseline_columns"])
    interaction = coded["interaction"]
    robust, ref = procedure(score_text)
    source_outcome = out_ok
    source_extra = extra_baseline
    source_interaction = coded["interaction_in_program"]
    source_site = "site" in score_text.lower()
    broad = all((out_ok, unrestricted, pop_ok, null_ok, site_ok, not extra_baseline, not interaction,
                 robust == "HC2", ref == "student_t", source_outcome, not source_extra, not source_interaction, source_site))
    exact_mean = bool(re.search(r"E\s*\[\s*Y[^]]*\|", model, re.I))
    normal_hits = positive_assumption(model, r"\bnormality\b|\bnormal[- ](?:error|disturbance|distribution)s?\b|\bgaussian\b")
    homosk_hits = positive_assumption(model, r"homosk|equal.{0,20}variance|constant.{0,20}variance")
    strict = broad and not exact_mean and not normal_hits and not homosk_hits
    ambiguity = []
    if broad and re.search(r"conditional mean", model, re.I) and not exact_mean:
        ambiguity.append("conditional-mean language not parsed as equality")
    if robust == "unclear" or ref == "unclear": ambiguity.append("source inference procedure unclear")
    if out_ok and not broad:
        failed = [k for k,v in {"unrestricted":unrestricted,"population":pop_ok,"null":null_ok,"site":site_ok,
                  "no_extra_baseline":not extra_baseline,"no_interaction":not interaction,"source_outcome":source_outcome,
                  "source_no_extra":not source_extra,"source_no_interaction":not source_interaction,
                  "hc2":robust=="HC2","student_t":ref=="student_t"}.items() if not v]
    else: failed = []
    return {"broad_common_procedure": broad, "strict_projection": strict,
            "exact_additive_conditional_mean": exact_mean,
            "positive_normality": normal_hits, "positive_homoskedasticity": homosk_hits,
            "robust_se": robust, "reference_distribution": ref,
            "legacy_code": coded, "failed_common_fields": failed, "ambiguity": ambiguity}


def roster_map(root, cohort):
    p = root / cohort / "roster.json" if (root/cohort/"roster.json").exists() else root / "input" / cohort / "roster.json"
    return {r["case_id"]: r for r in json.loads(p.read_text())["cases"]}


def score_path(root, cohort, record):
    compact=root/cohort/"scores"/f"{record['case_id']}.py"
    if compact.exists(): return compact
    base = root / "cohorts" / cohort / "originals" / record["model"]
    direct = base / f"{int(record['position']):04d}-{record['case_id'][:12]}" / "workspace/score.py"
    if direct.exists(): return direct
    found = list(base.glob(f"*-{record['case_id'][:12]}/workspace/score.py"))
    if len(found) != 1: raise FileNotFoundError((cohort, record["case_id"], found))
    return found[0]


def canonical_reference_scores(root, roster):
    gen_path = root/"reference/generate.py" if (root/"reference/generate.py").exists() else root / "input/reference/frozen/generate.py"
    gen = load_module(gen_path, "reference_generator")
    out = {}
    for cid, case in roster.items():
        data = gen.generate(int(case["seed"]), 2000, 0.0)
        cell, y, X = bank.design_cells(data)
        est, se, p = bank.full_score(cell, y, X)
        out[cid] = {"estimate": est, "std_error": se, "p_value": p}
    return out, sha256(gen_path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runtime_root", type=Path); ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args(); root = a.runtime_root.resolve(); a.out.mkdir(parents=True, exist_ok=True)
    rosters = {c: roster_map(root,c) for c in ("reference","anchor-a","anchor-b")}
    canonical, generator_sha = canonical_reference_scores(root, rosters["reference"])
    rows=[]
    for cohort in ("reference","anchor-a","anchor-b"):
        records=[]
        result_paths=[root/cohort/"records.jsonl"] if (root/cohort/"records.jsonl").exists() else list((root/"cohorts"/cohort/"node-results").glob("*.results.jsonl"))
        for p in result_paths:
            records.extend(json.loads(x) for x in p.read_text().splitlines())
        latest={}
        for r in records:
            if r.get("status") == "complete": latest[r["case_id"]] = r
        for cid,r in latest.items():
            sp=score_path(root,cohort,r); text=sp.read_text(errors="replace")
            labels=classify(r.get("commitment") or {},text); case=rosters[cohort][cid]
            row={"cohort":cohort,"case_id":cid,"position":r["position"],"seed":r.get("seed"),
                 "status":r["status"],"u":r.get("u"),"original_score":r.get("original_score"),
                 "commitment":r.get("commitment"),"final_report":r.get("final_report"),
                 "score_sha256":sha256(sp),**labels}
            if cohort=="reference":
                row["canonical_q_score"]=canonical[cid]
                row["selected_q_score_abs_difference"]=(None if not labels["broad_common_procedure"] or r.get("u") is None
                    else abs(float(r["u"])-canonical[cid]["p_value"]))
            else:
                row.update({k:case.get(k) for k in ("stratum","p_bin","reference_p","original_u","original_seed","tau")})
            rows.append(row)
    with (a.out/"scored_records.jsonl").open("w") as f:
        for r in sorted(rows,key=lambda x:(x["cohort"],x["position"])): f.write(json.dumps(r,sort_keys=True)+"\n")
    ambiguous=[r for r in rows if r["ambiguity"]]
    packet=[{"id":f"{r['cohort']}:{r['case_id']}","commitment":r["commitment"],
             "code_facts":{"robust_se":r["robust_se"],"reference_distribution":r["reference_distribution"],
                           "failed_common_fields":r["failed_common_fields"]}}
            for r in ambiguous]
    (a.out/"ambiguity_packet.json").write_text(json.dumps(packet,indent=2)+"\n")
    broad_rows=sorted((x for x in rows if x["broad_common_procedure"]),key=lambda x:(x["cohort"],x["position"]))
    broad_packet=[{"id":f"blind-{i:04d}","commitment":r["commitment"],
             "code_facts":{"outcome":r["legacy_code"]["outcome"],"followup":r["legacy_code"]["followup"],
                           "restricted":r["legacy_code"]["restricted"],"interaction":r["legacy_code"]["interaction"],
                           "baseline_columns":r["legacy_code"]["baseline_columns"],
                           "robust_se":r["robust_se"],"reference_distribution":r["reference_distribution"]}}
            for i,r in enumerate(broad_rows)]
    (a.out/"reader_candidates_blinded.json").write_text(json.dumps({"items":broad_packet},indent=2)+"\n")
    (a.out/"reader_private_map.json").write_text(json.dumps({f"blind-{i:04d}":{"cohort":r["cohort"],"case_id":r["case_id"],"position":r["position"]}
                                                       for i,r in enumerate(broad_rows)},indent=2)+"\n")
    summary={"records":len(rows),"complete_by_cohort":{c:sum(r["cohort"]==c for r in rows) for c in rosters},
             "broad_by_cohort":{c:sum(r["cohort"]==c and r["broad_common_procedure"] for r in rows) for c in rosters},
             "strict_by_cohort":{c:sum(r["cohort"]==c and r["strict_projection"] for r in rows) for c in rosters},
             "ambiguous":len(ambiguous),"reference_generator_sha256":generator_sha,
             "max_selected_q_score_abs_difference":max((r["selected_q_score_abs_difference"] for r in rows if r.get("selected_q_score_abs_difference") is not None),default=None)}
    (a.out/"collection_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
