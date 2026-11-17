#!/usr/bin/env python3
"""Tool-disabled Sol reader sharing the originals' auth, stops, and 12-point guard."""
import argparse,json,os,shutil,sys,tarfile,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(os.environ["CCS_ROOT"]);REVIEW=Path(os.environ["CCS_REVIEW"]);CONTROL=Path(os.environ.get("CCS_CONTROL",ROOT/"control"));sys.path[:0]=[str(ROOT/"runtime/established")]
os.environ["MMO_ROOT"]=str(ROOT);os.environ["MMO_FROZEN"]=str(ROOT/"frozen");os.environ["CCS_CONTROL"]=str(CONTROL)
import common,run_codex
SCHEMA=json.loads((ROOT/"scoring/reader_schema.json").read_text());OUT=REVIEW/"out"

class Route(run_codex.CodexRoute):
 def config_text(self):
  x=super().config_text()
  for k in ("features.shell_tool","features.unified_exec","features.code_mode_host"):x=x.replace(f"{k} = true",f"{k} = false")
  return x

def archive(c,route,case,attempt):
 route.cleanup(c)
 for p in c.rglob("*"):
  if not p.is_file() or p.stat().st_size>20_000_000:continue
  try:text=p.read_text()
  except (UnicodeDecodeError,OSError):continue
  for secret in route.secrets():text=text.replace(secret,"<redacted>")
  p.write_text(text)
 job=os.getenv("SLURM_JOB_ID","local");task=os.getenv("SLURM_ARRAY_TASK_ID","0");dest=REVIEW/"archives"/f"{case['id']}-job{job}-task{task}-a{attempt}.tar.gz";dest.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(dest,"x:gz") as t:t.add(c,arcname="receipt")
 return str(dest)

def one(case,route,work,result,stop):
 if stop.exists() or route.paused:return
 for attempt in range(1,common.MAX_ATTEMPTS+1):
  c=work/f"{case['id']}-a{attempt}"
  for n in list(common.GUEST)+["logs"]:(c/n).mkdir(parents=True,exist_ok=True)
  route.prepare(c);rec={k:case[k] for k in ("id","task","read","position")};rec.update(attempt=attempt,reviewer=route.model)
  try:
   x=route.phase(c,"analysis",case["prompt"],schema=SCHEMA);rec.update({k:x.get(k) for k in ("ok","fault_kind","error","usage","seconds")})
   if x.get("ok"):rec["label"]=json.loads(x["answer"])
  except Exception as e:rec.update(ok=False,fault_kind="runner",error=f"{type(e).__name__}: {e}")
  finally:route.cleanup(c)
  try:rec.update(route.facts(c))
  except Exception as e:rec["facts_error"]=f"{type(e).__name__}: {e}"
  try:rec["receipt_archive"]=archive(c,route,case,attempt)
  except Exception as e:rec["archive_error"]=f"{type(e).__name__}: {e}"
  line=json.dumps(rec)
  for s in route.secrets():line=line.replace(s,"<redacted>")
  with common.SINK_LOCK,result.open("a") as f:f.write(line+"\n")
  shutil.rmtree(c,ignore_errors=True)
  if rec.get("label") or route.paused or stop.exists():return
  time.sleep(20*attempt)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--account",required=True,choices=("primary","second"));ap.add_argument("--task",type=int,default=int(os.getenv("SLURM_ARRAY_TASK_ID",0)));ap.add_argument("--tasks",type=int,default=1);ap.add_argument("--workers",type=int,default=12);a=ap.parse_args()
 stop=CONTROL/f"STOP-{a.account}";lane=ROOT/"review-lanes"/a.account;lane.mkdir(parents=True,exist_ok=True);link=lane/"STOP-codex"
 if not link.exists() and not link.is_symlink():link.symlink_to(stop)
 if stop.exists():raise SystemExit(f"shared account stop exists: {stop}")
 common.RUN=lane;route=Route("gpt-6.1-sol",a.account);OUT.mkdir(parents=True,exist_ok=True);cases=common.read_jsonl(REVIEW/"cases.jsonl");done=set()
 for p in OUT.glob("results-*.jsonl"):done|={r["id"] for r in common.read_jsonl(p) if r.get("label")}
 cases=[c for c in cases[a.task::a.tasks] if c["id"] not in done];work=Path(os.getenv("TMPDIR","/tmp"))/f"ccs-review-{os.getenv('SLURM_JOB_ID','local')}-{a.task}";work.mkdir(parents=True,exist_ok=True);result=OUT/f"results-{os.getenv('SLURM_JOB_ID','local')}-{a.task}.jsonl"
 with ThreadPoolExecutor(a.workers) as pool:list(pool.map(lambda c:one(c,route,work,result,stop),cases))
 print(json.dumps({"assigned":len(cases),"paused":route.paused}))

if __name__=="__main__":main()
