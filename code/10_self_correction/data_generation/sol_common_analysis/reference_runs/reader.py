#!/usr/bin/env python3
"""Sequential blinded readers under the scientific run's shared 77% meter guard."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys, time

sys.path.insert(0, str(Path(__file__).resolve().parent / "established"))
import common
from run_codex import CodexRoute

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class ToolDisabledRoute(CodexRoute):
    def config_text(self):
        disabled = {"features.code_mode_host", "features.shell_tool", "features.unified_exec"}
        lines = [x for x in super().config_text().splitlines() if x.split(" =", 1)[0] not in disabled]
        lines += [f"{x} = false" for x in sorted(disabled)]
        return "\n".join(lines) + "\n"

ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); a=ap.parse_args(); root=a.root.resolve()
inp=root/"reader-input"; out=root/"readers"; out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((inp/"reader_batches/MANIFEST.json").read_text())
assert sha(inp/"READER_PROMPT_FROZEN.txt")==manifest["prompt_sha256"]
assert sha(inp/"READER_SCHEMA.json")==manifest["schema_sha256"]
assert manifest["batches"]==10 and manifest["items"]==193
for name,want in manifest["batch_sha256"].items(): assert sha(inp/"reader_batches"/name)==want,name
stop=root/"stops/primary"; run_stop=out/"STOP-codex"
if not run_stop.exists() and not run_stop.is_symlink():
    try: run_stop.symlink_to(stop)
    except FileExistsError: pass
route=ToolDisabledRoute("gpt-6.1-sol","primary")
schema=json.loads((inp/"READER_SCHEMA.json").read_text()); base=(inp/"READER_PROMPT_FROZEN.txt").read_text()
task=os.environ.get("READER_TASK")
indices=[int(task)] if task is not None else list(range(10))
for i in indices:
    batch=f"batch-{i:02d}"
    saved=out/batch
    result_path=saved/"result.json"
    if result_path.exists(): continue
    if stop.exists() or route.paused: break
    case=Path(os.environ.get("TMPDIR","/tmp"))/f"reader-{os.getpid()}-{batch}"
    for name in list(common.GUEST)+["logs"]: (case/name).mkdir(parents=True,exist_ok=True)
    route.prepare(case); started=time.time()
    prompt=base+"\n\nBLINDED BATCH JSON:\n"+(inp/"reader_batches"/f"{batch}.json").read_text()
    rec={"batch":batch,"started":started,"input_sha256":manifest["batch_sha256"][f"{batch}.json"]}
    try:
        phase=route.phase(case,"analysis",prompt,schema=schema); rec["phase"]=phase
        if phase.get("ok"):
            answer=json.loads(phase["answer"]); rec["answer"]=answer
            rec["item_count"]=len(answer.get("items",[]))
        rec.update(route.facts(case))
    except Exception as e:
        rec["runner_error"]=f"{type(e).__name__}: {e}"
    finally:
        route.cleanup(case); rec["finished"]=time.time()
    saved.mkdir(exist_ok=True)
    shutil.copytree(case/"logs",saved/"logs",dirs_exist_ok=True)
    if (case/"codex-home/sessions").exists(): shutil.copytree(case/"codex-home/sessions",saved/"sessions",dirs_exist_ok=True)
    result_path.write_text(json.dumps(rec,indent=2)+"\n")
    shutil.rmtree(case,ignore_errors=True)
    used=[float(w["used_percent"]) for w in rec.get("meter_windows",{}).values() if w.get("used_percent") is not None]
    if any(x>=77 for x in used):
        if not stop.exists(): stop.write_text(json.dumps({"reason":"reader meter guard","threshold":77,"meter_windows":rec.get("meter_windows"),"time":time.time()}))
        break
print(json.dumps({"completed_batches":len(list(out.glob("batch-??/result.json"))),"stopped":stop.exists()}))
