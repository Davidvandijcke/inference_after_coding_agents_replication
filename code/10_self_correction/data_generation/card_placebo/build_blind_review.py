#!/usr/bin/env python3
"""Build immutable opaque Card report reads from completed records. No model calls."""
import argparse, hashlib, io, json, re, secrets, tarfile
from pathlib import Path


def read_records(roots):
    kept = {}
    for root in roots:
        for path in list(root.glob("node-results/*.results.jsonl")) + list(root.glob("lanes/*/node-results/*.results.jsonl")):
            for line in path.read_text(errors="replace").splitlines():
                try: r = json.loads(line)
                except ValueError: continue
                cid = r.get("case_id")
                if cid and (cid not in kept or r.get("status") == "complete"): kept[cid] = r
    return kept


def full_reports(roots, records):
    wanted=set(records);linked={cid:set(re.findall(r"(?:/workspace/)?([A-Za-z0-9_.-]+\.md)",r.get("final_report", ""))) for cid,r in records.items()}
    excluded={"CODEBOOK.md","DESIGN.md","AGENTS.md","README.md"}
    out={}
    for root in roots:
        archives=list(root.glob("node-archives/*.tar"))+list(root.glob("lanes/*/node-archives/*.tar"))
        for outer_path in archives:
            with tarfile.open(outer_path) as outer:
                for member in outer.getmembers():
                    if not member.name.endswith(".tar.gz"):continue
                    raw=outer.extractfile(member).read()
                    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as inner:
                        names=inner.getnames();cid=next((c for c in wanted if any(n.startswith(c+"/") for n in names)),None)
                        if not cid:continue
                        parts=[]
                        for name in names:
                            base=name.rsplit("/",1)[-1]
                            if base not in excluded and (base in ("report.md","analysis_report.md","analysis_record.md") or base in linked.get(cid,set())):
                                f=inner.extractfile(name)
                                if f:parts.append(f.read().decode("utf-8","replace"))
                        if parts:out[cid]="\n\n".join(parts)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--record-root",type=Path,action="append",required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--exclude-key",type=Path,action="append",default=[]); ap.add_argument("--limit",type=int); a=ap.parse_args()
    cases_path,key_path=a.out/"cases.jsonl",a.out/"private/key.json"
    if cases_path.exists() or key_path.exists(): raise SystemExit("refusing to overwrite blind packet")
    prompt=Path(__file__).with_name("reader_prompt.txt").read_text(); salt=secrets.token_hex(32); cases=[]; key={};excluded=set();all_records=read_records(a.record_root);artifacts=full_reports(a.record_root,all_records)
    for p in a.exclude_key:excluded.update(x["case_id"] for x in json.loads(p.read_text()).values())
    eligible=[(cid,r) for cid,r in sorted(all_records.items()) if r.get("status")=="complete" and r.get("final_report") and cid not in excluded]
    if a.limit is not None:eligible=eligible[:a.limit]
    for cid,r in eligible:
        opaque=hashlib.sha256(f"{salt}:{cid}".encode()).hexdigest()
        cases.append({"id":opaque,"task":"card_facts","read":1,"prompt":prompt.replace("{{REPORT}}",r["final_report"]).replace("{{FULL_REPORT}}",artifacts.get(cid,"")).replace("{{DECLARATION}}",json.dumps(r.get("commitment") or {},ensure_ascii=False,indent=2))})
        key[opaque]={"case_id":cid,"model":r.get("model"),"slot":r.get("slot")}
    cases.sort(key=lambda x:x["id"])
    for i,c in enumerate(cases): c["position"]=i
    (a.out/"private").mkdir(parents=True,exist_ok=True); (a.out/"private").chmod(0o700)
    cases_path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in cases)); key_path.write_text(json.dumps(key,indent=1)); key_path.chmod(0o600)
    print(json.dumps({"reports":len(cases),"with_full_report_artifact":sum(key[x["id"]]["case_id"] in artifacts for x in cases),"excluded_prior":len(excluded),"faults_or_prior_excluded":len(all_records)-len(cases)}))


if __name__=="__main__": main()
