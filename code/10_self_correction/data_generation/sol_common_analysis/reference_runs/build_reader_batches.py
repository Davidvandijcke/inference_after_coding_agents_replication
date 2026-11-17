#!/usr/bin/env python3
"""Freeze blinded broad-candidate reader batches plus deterministic old controls."""
import hashlib, importlib.util, json
from pathlib import Path

HERE=Path(__file__).resolve().parent; BASE=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location("collector",HERE/"collect_and_score.py");c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
new=json.loads((HERE/"reader_candidates_blinded.json").read_text())["items"]
private=json.loads((HERE/"reader_private_map.json").read_text())
old=[json.loads(x) for x in (BASE/"audit/membership.jsonl").open()]
controls=[]
for kind,want_strong in (("weak",False),("strong",True)):
    eligible=[]
    for r in sorted(old,key=lambda x:(x["cohort"],x["slot"])):
        if not r["canonical_primary_member"]:continue
        strong="E[Y" in (r["commitment_model"] or "") and "does not impose" not in (r["commitment_model"] or "")
        if strong!=want_strong:continue
        sp=BASE/"audit/source"/r["cohort"]/"originals"/r["model"]/f"{r['slot']:02d}"/"workspace/score.py"
        cm={"model":r["commitment_model"],"null":r["commitment_null"],"population":r["commitment_population"],"outcome":r["commitment_outcome"],"sample_restrictions":r["commitment_sample_restrictions"]}
        lab=c.classify(cm,sp.read_text())
        eligible.append((r,cm,lab))
    for j,(r,cm,lab) in enumerate(eligible[:10]):
        oid=f"control-{kind}-{j:02d}"
        controls.append({"id":oid,"commitment":cm,"code_facts":{"outcome":lab["legacy_code"]["outcome"],"followup":lab["legacy_code"]["followup"],"restricted":lab["legacy_code"]["restricted"],"interaction":lab["legacy_code"]["interaction"],"baseline_columns":lab["legacy_code"]["baseline_columns"],"robust_se":lab["robust_se"],"reference_distribution":lab["reference_distribution"]}})
        private[oid]={"control":kind,"cohort":r["cohort"],"slot":r["slot"],"expected":"MATCH" if kind=="weak" else "NONMATCH"}
items=new+controls
if len(items)>240: raise SystemExit(f"reader capacity exceeded: {len(items)}")
out=HERE/"reader_batches";out.mkdir(exist_ok=True)
batch_files=[]
for i in range(0,len(items),20):
    p=out/f"batch-{i//20:02d}.json";p.write_text(json.dumps({"items":items[i:i+20]},indent=2)+"\n");batch_files.append(p)
(out/"PRIVATE_MAP.json").write_text(json.dumps(private,indent=2,sort_keys=True)+"\n")
manifest={"items":len(items),"new_broad_candidates":len(new),"old_controls":len(controls),"batches":len(batch_files),"batch_sha256":{p.name:sha(p) for p in batch_files},"prompt_sha256":sha(HERE/"READER_PROMPT_FROZEN.txt"),"schema_sha256":sha(HERE/"READER_SCHEMA.json"),"private_map_sha256":sha(out/"PRIVATE_MAP.json")}
(out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps(manifest,indent=2))
