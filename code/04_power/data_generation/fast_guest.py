"""Inside the scoring sandbox: run the frozen score.py on every dataset in a folder.

The program runs unchanged, as __main__, with the same arguments. The libraries it imports
are loaded once, and each dataset gets a forked child, so the cost of starting Python is
paid once per sandbox instead of once per dataset. Failure handling matches
runner.score_dataset: whatever output file exists is read, and anything else is no result.

Usage (guest): python3 fast_guest.py DATA_DIR OUT_DIR
"""
import importlib
import json
import os
import re
import runpy
import signal
import sys

data_dir, out_dir = sys.argv[1], sys.argv[2]
source = open("/workspace/score.py", errors="replace").read()
names = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", source, re.M))
names |= {f"{m}.{s}" for m, s in re.findall(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+([A-Za-z_]\w*)", source, re.M)}
for name in sorted(names):
    try:
        importlib.import_module(name)
    except Exception:
        pass
sys.path.insert(0, "/workspace")
results = {}
for f in sorted(os.listdir(data_dir)):
    j = f.split("-")[1].split(".")[0]
    out = f"{out_dir}/out-{j}.json"
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, 1)
            os.dup2(null, 2)
            signal.alarm(300)
            sys.argv = ["score.py", "--data", f"{data_dir}/{f}", "--output", out]
            runpy.run_path("/workspace/score.py", run_name="__main__")
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        except BaseException:
            code = 1
        os._exit(code)
    os.waitpid(pid, 0)
    try:
        results[j] = json.load(open(out))
    except (OSError, ValueError):
        results[j] = None
    if os.path.exists(out):
        os.unlink(out)
with open(f"{out_dir}/results.json", "w") as handle:
    json.dump(results, handle)
