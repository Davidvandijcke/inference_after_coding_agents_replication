#!/usr/bin/env python3
"""Validate frozen reader outputs, controls, IDs, and attach labels to records."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
batch_dir=HERE/"reader_batches"; result_dir=HERE/"reader_results/readers"
private=json.loads((batch_dir/"PRIVATE_MAP.json").read_text())
seen={}; receipts=[]
allowed_labels={"MATCH","NONMATCH","AMBIGUOUS"}; allowed_stronger={"NONE","EXACT_CONDITIONAL_MEAN","NORMALITY","HOMOSKEDASTICITY","OTHER"}
for bf in sorted(batch_dir.glob("batch-*.json")):
    expected=json.loads(bf.read_text())["items"]
    rf=result_dir/bf.stem/"result.json"; rec=json.loads(rf.read_text())
    assert rec["input_sha256"]==hashlib.sha256(bf.read_bytes()).hexdigest()
    assert rec["session_models"]==["gpt-6.1-sol"] and rec["session_efforts"]==["high"]
    answer=json.loads(rec["answer"]) if isinstance(rec["answer"],str) else rec["answer"]
    got=answer["items"]; assert len(got)==len(expected)==rec["item_count"]
    assert [x["id"] for x in got]==[x["id"] for x in expected]
    for x in got:
        assert set(x)=={"id","label","reason","stronger_assumption"}
        assert x["id"] not in seen and x["id"] in private
        assert x["label"] in allowed_labels and x["stronger_assumption"] in allowed_stronger
        assert isinstance(x["reason"],str) and x["reason"].strip()
        seen[x["id"]]=x
    receipts.append({"batch":bf.stem,"items":len(got),"input_sha256":rec["input_sha256"],"meter_windows":rec.get("meter_windows"),"finished":rec.get("finished")})
assert set(seen)==set(private)
controls=[]
for oid,meta in private.items():
    if "expected" in meta:
        controls.append({"id":oid,"expected":meta["expected"],"observed":seen[oid]["label"],"pass":seen[oid]["label"]==meta["expected"]})
assert len(controls)==20
rows=[json.loads(x) for x in (HERE/"scored_records.jsonl").open()]
lookup={(r["cohort"],r["case_id"]):r for r in rows}
for oid,label in seen.items():
    meta=private[oid]
    if "expected" in meta:continue
    r=lookup[(meta["cohort"],meta["case_id"])]
    r["reader_label"]=label["label"];r["reader_reason"]=label["reason"];r["reader_stronger_assumption"]=label["stronger_assumption"]
    r["strict_reader_match"]=label["label"]=="MATCH"
with (HERE/"scored_records_reader.jsonl").open("w") as f:
    for r in sorted(rows,key=lambda x:(x["cohort"],x["position"])):f.write(json.dumps(r,sort_keys=True)+"\n")
summary={"items":len(seen),"new_candidates":len(seen)-20,"controls":controls,
 "control_accuracy":sum(x["pass"] for x in controls)/20,"control_errors":[x for x in controls if not x["pass"]],
 "new_labels":{k:sum(x["label"]==k for oid,x in seen.items() if "expected" not in private[oid]) for k in sorted(allowed_labels)},
 "receipts":receipts}
(HERE/"READER_VALIDATION.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
with (HERE/"reader_labels.jsonl").open("w") as f:
    for oid,x in sorted(seen.items()):f.write(json.dumps({"id":oid,**private[oid],**x},sort_keys=True)+"\n")
print(json.dumps(summary,indent=2))
