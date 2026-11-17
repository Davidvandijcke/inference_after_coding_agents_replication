"""Shared pieces of the multi-model originals runners.

Design: docs/MULTI_MODEL_ORIGINALS_DESIGN_2026-10-30.md. The guest layout, environment, data scope,
scoring sandbox and file-light storage are the main run's
(open-ended-sim-redesign-20261026/pilot/source/runner.py). What differs by route is in run_codex.py
and run_claude.py, each of which supplies a Route.
"""
import collections
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import tarfile
import threading
import time

HERE = Path(__file__).resolve().parent
FROZEN = Path(os.environ.get("MMO_FROZEN", HERE / "frozen"))
FROZEN_FILES = ("CODEBOOK.md", "research_prompt.txt", "commitment_prompt.txt", "generate.py")
RUN = Path(os.environ.get("MMO_ROOT", "<cluster-scratch>/multi-model-originals-20261030"))
MAIN_RUN = Path("<cluster-scratch>/open-ended-sim-redesign-20261026")  # read only
CURATED_PYTHON = Path("<cluster-scratch>/immigration-scientific-python-v1")
MODELS = ("gpt-6.1-terra", "gpt-6.1-sol", "claude-opus-5-5", "claude-sonnet-5-5")
EFFORT = "high"
PROXY = "<cluster-proxy>"
NO_PROXY = "localhost,127.0.0.1,::1"
GUEST = {"workspace": "/workspace", "home": "/home/researcher", "tmp": "/tmp"}
GUEST_SCHEMA = "/run/scientific/schema.json"
PUBLIC_ETC = ["/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/passwd", "/etc/group",
              "/etc/localtime", "/etc/ssl", "/etc/pki", "/etc/ld.so.cache", "/etc/fonts", "/etc/os-release",
              "/etc/services", "/etc/protocols", "/etc/alternatives"]
DATA_SCOPE = ("Use the supplied packet as the sole research data source. Internet access may be used for "
              "technical documentation, software and package installation. Do not replace or augment research "
              "observations with downloaded datasets. Keep supplied packet files unchanged and write analysis "
              "artifacts separately.")
FIELDS = ("model", "null", "population", "outcome", "sample_restrictions")
COMMITMENT_SCHEMA = {"type": "object", "properties": {x: {"type": "string"} for x in FIELDS},
                     "required": list(FIELDS), "additionalProperties": False}
TIMEOUT = {"analysis": int(os.environ.get("MMO_PHASE_TIMEOUT", 3600)), "commitment": int(os.environ.get("MMO_PHASE_TIMEOUT", 900))}
MAX_ATTEMPTS = int(os.environ.get("MMO_MAX_ATTEMPTS", 3))
FILE_QUOTA_STOP = 0.97
MAX_ARCHIVED_FILE = 20_000_000
SINK_LOCK = threading.Lock()
QUEUE_LOCK = threading.Lock()


def guest_env(agent_dir):
    """The main run's guest environment, with the agent binary's folder first on PATH."""
    return {
        "PATH": f"{agent_dir}:<cluster-software>/arc/python/3.12.1/bin:<cluster-software>/arc/stacks/gcc/13.2.0/R/4.5.1/bin:"
                "<cluster-software>/coe/o/image-libraries/220318/bin:<cluster-software>/arc/gcc/13.2.0/bin:/usr/local/bin:/usr/bin",
        "LD_LIBRARY_PATH": "<cluster-software>/arc/python/3.12.1/lib:<cluster-software>/arc/stacks/gcc/13.2.0/R/4.5.1/lib64/R/lib:"
                           "<cluster-software>/coe/o/image-libraries/220318/lib:<cluster-software>/arc/gcc/13.2.0/lib64",
        "LIBRARY_PATH": "<cluster-software>/arc/gcc/13.2.0/lib64",
        "CPATH": "<cluster-software>/arc/gcc/13.2.0/lib/gcc/x86_64-pc-linux-gnu/13.2.0/include-fixed:"
                 "<cluster-software>/arc/gcc/13.2.0/lib/gcc/x86_64-pc-linux-gnu/13.2.0/include",
        "PKG_CONFIG_PATH": "<cluster-software>/arc/stacks/gcc/13.2.0/R/4.5.1/lib64/pkgconfig:"
                           "<cluster-software>/coe/o/image-libraries/220318/lib/pkgconfig",
        "PYTHONPATH": "/opt/scientific-python", "PYTHONDONTWRITEBYTECODE": "1",
        "R_LIBS_SITE": "<cluster-software>/arc/stacks/gcc/13.2.0/Rtidyverse/4.5.1/2025-09-30",
        "R_ENVIRON_USER": "/dev/null", "R_PROFILE_USER": "/dev/null", "R_ENVIRON": "/dev/null", "R_PROFILE": "/dev/null",
        "PYTHONNOUSERSITE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "HTTPS_PROXY": PROXY, "HTTP_PROXY": PROXY, "https_proxy": PROXY, "http_proxy": PROXY,
        "NO_PROXY": NO_PROXY, "no_proxy": NO_PROXY,
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


def read_jsonl(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except ValueError:  # a line cut by a kill
            continue
    return rows


# ---------------------------------------------------------------- sandbox

def public_mounts():
    args = ["--dir", "/etc", "--dir", "/opt", "--dir", "/run"]
    for top in ("/usr", "/sw"):
        args += ["--ro-bind", top, top]
    for link in ("/bin", "/sbin", "/lib", "/lib64"):
        if os.path.islink(link):
            args += ["--symlink", os.readlink(link), link]
        elif os.path.isdir(link):
            args += ["--ro-bind", link, link]
    for f in PUBLIC_ETC:
        if os.path.exists(f):
            args += ["--ro-bind", f, f]
    return args + ["--ro-bind", str(CURATED_PYTHON), "/opt/scientific-python"]


def bwrap(case_dir, guest_cmd, *, env, network=True, binds=None, ro_binds=None, schema=None, dirs=(),
          cwd=GUEST["workspace"]):
    """The main run's guest: an empty root with the public software, the case's own folders and nothing else."""
    c = Path(case_dir)
    args = ["/usr/bin/bwrap", "--die-with-parent", "--unshare-user", "--unshare-pid", "--unshare-ipc", "--new-session"]
    if not network:
        args.append("--unshare-net")
    args += ["--tmpfs", "/", "--dev", "/dev", "--proc", "/proc"] + public_mounts()
    for name, guest in GUEST.items():
        args += ["--bind", str(c / name), guest]
    for guest in dirs:
        args += ["--dir", guest]
    for guest, host in (ro_binds or {}).items():
        args += ["--ro-bind", str(host), guest]
    if (c / "packet").is_dir():
        for f in sorted((c / "packet").iterdir()):
            args += ["--ro-bind", str(f), f"{GUEST['workspace']}/{f.name}"]
    if schema is not None:
        args += ["--dir", "/run/scientific", "--ro-bind", str(schema), GUEST_SCHEMA]
    for guest, host in (binds or {}).items():
        args += ["--dir", guest, "--bind", str(host), guest]
    for k, val in sorted(env.items()):
        args += ["--setenv", k, val]
    return args + ["--chdir", cwd, "--"] + list(guest_cmd)


def run_guest(case_dir, phase, cmd, prompt, secret_env=None):
    """One agent turn. Secrets reach the guest through the process environment, never the command line."""
    log = Path(case_dir) / "logs"
    t0 = time.time()
    timed_out = False
    with open(log / f"{phase}.events.jsonl", "w") as out, open(log / f"{phase}.stderr", "w") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
                                env={"PATH": "/usr/bin:/bin", **(secret_env or {})})
        try:
            proc.communicate(prompt, timeout=TIMEOUT[phase])
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
    return {"phase": phase, "seconds": round(time.time() - t0, 1), "returncode": proc.returncode, "timed_out": timed_out}


def parse_commitment(text):
    """The main run's parse, then the five declared string fields."""
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start:end + 1] if start >= 0 and end > start else text)
    if not isinstance(obj, dict) or set(obj) != set(FIELDS) or not all(isinstance(v, str) and v.strip() for v in obj.values()):
        raise ValueError("commitment does not have the five declared fields")
    return obj


# ---------------------------------------------------------------- scoring, as in the main run

def score_dataset(workspace, data_path, out_dir):
    """Run the agent's score.py on one dataset with no network and a scratch copy of its workspace."""
    run = Path(out_dir)
    for name in GUEST:
        (run / name).mkdir(parents=True, exist_ok=True)
    for f in Path(workspace).iterdir():
        if f.name == "data.csv":
            continue
        if f.is_file() and f.stat().st_size < MAX_ARCHIVED_FILE:
            shutil.copy2(f, run / "workspace" / f.name)
        elif f.is_dir() and f.name not in ("__pycache__",):
            shutil.copytree(f, run / "workspace" / f.name, symlinks=True,
                            ignore=shutil.ignore_patterns("*.csv", "*.parquet", "*.rds", "*.RData"))
    (run / "scoreout").mkdir(exist_ok=True)
    cmd = bwrap(run, ["python3", "score.py", "--data", "/scoredata/data.csv", "--output", "/scoreout/out.json"],
                env=dict(guest_env("/opt/agent"), HOME=GUEST["home"], TMPDIR=GUEST["tmp"]), network=False,
                dirs=("/scoredata",), ro_binds={"/scoredata/data.csv": data_path}, binds={"/scoreout": run / "scoreout"})
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env={"PATH": "/usr/bin:/bin"})
        out = run / "scoreout" / "out.json"
        try:
            res = json.loads(out.read_text()) if out.exists() else None
        except ValueError:  # malformed out.json: a failed score, handled like a crash
            res = None
        return {"returncode": p.returncode, "result": res, "stderr": p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "result": None, "stderr": "timeout"}


def p_from(result):
    try:
        p = float(result["p_value"])
        return p if 0 <= p <= 1 else None
    except (TypeError, KeyError, ValueError):
        return None


# ---------------------------------------------------------------- cases

def load_kept():
    """One kept record per case: a complete record over anything else, then the later file (DUP-01)."""
    out = {}
    for p in sorted((RUN / "node-results").glob("*.results.jsonl")):
        for rec in read_jsonl(p):
            prev = out.get(rec["case_id"])
            if prev is None or rec.get("status") == "complete" or prev.get("status") != "complete":
                out[rec["case_id"]] = rec
    return out


def classify(rec, route):
    """A refused, cut-off or timed-out turn is a pipeline fault, never an agent outcome."""
    if rec.get("runner_error"):
        return "fault", "runner"
    for name in ("analysis", "commitment"):
        phase = rec["phases"].get(name)
        if phase is None:
            if name == "analysis":
                return "fault", "no_phase"
            continue
        if not phase.get("ok"):
            return "fault", phase.get("fault_kind") or ("timeout" if phase.get("timed_out") else "cut_off")
    if rec.get("session_models") and rec["session_models"] != [route.model]:
        return "fault", "wrong_model"
    return "complete", None


def scrub_case(case_dir, secrets):
    """Replace any secret that reached a saved file. A check, not the protection: the agent's shell
    never holds the token, and this counts any file where one appears anyway."""
    n = 0
    for f in Path(case_dir).rglob("*"):
        if not f.is_file() or f.is_symlink() or f.stat().st_size > MAX_ARCHIVED_FILE:
            continue
        b = f.read_bytes()
        hit = False
        for s in secrets:
            if s.encode() in b:
                b, hit = b.replace(s.encode(), b"<redacted>"), True
        if hit:
            f.write_bytes(b)
            n += 1
    return n


def sink(rec, case_dir, route, campaign):
    """Append the attempt to this node's one results file and one archive."""
    c = Path(case_dir)
    packed = c.parent / f"{rec['case_id']}-a{rec['attempt']}.tar.gz"
    keep = [c / "logs", c / "workspace"] + route.keep(c)

    def small(ti):
        return None if ti.size > MAX_ARCHIVED_FILE or ti.name.endswith("auth.json") else ti

    with tarfile.open(packed, "w:gz") as t:
        for p in keep:
            if p.exists():
                t.add(p, arcname=f"{rec['case_id']}/{p.relative_to(c)}", filter=small)
    line = json.dumps(rec)
    for s in route.secrets():
        line = line.replace(s, "<redacted>")
    with SINK_LOCK:
        with open(RUN / "node-results" / f"{campaign}.results.jsonl", "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        with tarfile.open(RUN / "node-archives" / f"{campaign}.tar", "a") as t:
            t.add(packed, arcname=packed.name)
    packed.unlink()


def freeze_workspace(c, case, rec):
    """Keep the deliverable for scoring and run it on the original dataset, as the main run did."""
    frozen = RUN / "originals" / rec["model"] / f"{case['slot']:02d}" / "workspace"
    if frozen.exists():
        shutil.rmtree(frozen)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(c / "workspace", frozen, symlinks=True, ignore=shutil.ignore_patterns("data.csv", "CODEBOOK.md"))
    s = score_dataset(frozen, str(c / "packet" / "data.csv"), c / "score-original")
    rec["original_score"] = s
    rec["u"] = p_from(s["result"])


def run_attempt(case, route, work_root, attempt, campaign):
    cid = case["case_id"]
    c = Path(work_root) / f"{cid}-a{attempt}"
    for name in list(GUEST) + ["packet", "logs"]:
        (c / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN / "data" / "originals" / case["data"], c / "packet" / "data.csv")
    shutil.copy2(FROZEN / "CODEBOOK.md", c / "packet" / "CODEBOOK.md")
    route.prepare(c)
    rec = {"case_id": cid, "model": case["model"], "slot": case["slot"], "seed": case["seed"], "position": case["position"],
           "route": route.label, "attempt": attempt, "effort": EFFORT, "started": time.time(), "host": socket.gethostname(),
           "data_sha256": sha(c / "packet" / "data.csv"), "phases": {}}
    try:
        a = route.phase(c, "analysis", (FROZEN / "research_prompt.txt").read_text())
        rec["phases"]["analysis"] = a
        if a.get("ok"):
            rec["final_report"] = a["answer"][:20000]
            cm = route.phase(c, "commitment", (FROZEN / "commitment_prompt.txt").read_text(),
                             session_id=a["session_id"], schema=COMMITMENT_SCHEMA)
            rec["phases"]["commitment"] = cm
            if cm.get("ok"):
                try:
                    rec["commitment"] = parse_commitment(cm["answer"])
                except (ValueError, TypeError) as e:
                    rec["commitment_error"] = str(e)
        rec["has_score_py"] = (c / "workspace" / "score.py").exists()
    except Exception as e:  # logged as a fault
        rec["runner_error"] = f"{type(e).__name__}: {e}"
    finally:
        route.cleanup(c)
        rec["finished"] = time.time()
    try:
        rec.update(route.facts(c))
    except Exception as e:
        rec["facts_error"] = f"{type(e).__name__}: {e}"
    rec["data_unchanged"] = sha(c / "packet" / "data.csv") == rec["data_sha256"]
    rec["status"], rec["fault_kind"] = classify(rec, route)
    if rec["status"] == "complete" and rec.get("has_score_py"):
        try:
            freeze_workspace(c, case, rec)
        except Exception as e:
            rec["freeze_error"] = f"{type(e).__name__}: {e}"
    rec["token_scrubbed_files"] = scrub_case(c, route.secrets())
    sink(rec, c, route, campaign)
    shutil.rmtree(c, ignore_errors=True)
    return rec


def file_quota_fraction():
    out = subprocess.run(["scratch-quota", "<quota-group>"], capture_output=True, text=True, timeout=120).stdout
    for line in out.splitlines():
        if "FILESET" in line and "|" in line:
            nums = [x for x in line.split("|")[1].split() if x.isdigit()]
            if len(nums) >= 3 and int(nums[2]) > 0:
                return int(nums[0]) / int(nums[2])
    return None


def file_quota_guard(stop):
    while True:
        try:
            frac = file_quota_fraction()
            if frac is not None and frac >= FILE_QUOTA_STOP and not stop.exists():
                stop.write_text(json.dumps({"reason": "scratch file-count quota", "fraction": frac, "time": time.time()}))
        except Exception:
            pass
        time.sleep(300)


def record_versions(route, campaign, work_root):
    """Versions from inside the guest plus hashes of the frozen files, once per node."""
    run = Path(work_root) / "version-check"
    for name in GUEST:
        (run / name).mkdir(parents=True, exist_ok=True)
    rec = {"campaign": campaign, "host": socket.gethostname(), "time": time.time(), "model": route.model,
           "effort": EFFORT, "route": route.label, "agent_sha256": route.binary_sha(),
           "files": {f: sha(FROZEN / f) for f in FROZEN_FILES}}
    for key, guest_cmd in route.version_commands().items():
        p = subprocess.run(bwrap(run, guest_cmd, env=route.guest_env(), network=False, ro_binds=route.ro_binds(),
                                 dirs=route.dirs()), capture_output=True, text=True, timeout=120,
                           env={"PATH": "/usr/bin:/bin"})
        rec[key] = (p.stdout + p.stderr).strip()[:300]
    write_json(RUN / "node-status" / f"{campaign}.versions.json", rec)


def run_cases(route, task, tasks, workers, campaign_name, limit=None):
    """Run this task's pending cases of one model, in roster order, with staggered workers."""
    roster = [c for c in json.loads((RUN / "roster.json").read_text())["cases"] if c["model"] == route.model]
    roster.sort(key=lambda c: c["position"])
    if limit:
        roster = roster[:limit]
    kept = load_kept()
    pending = [c for c in roster if kept.get(c["case_id"], {}).get("status") != "complete"][task::tasks]
    job = os.environ.get("SLURM_JOB_ID", "local")
    campaign = f"{campaign_name}-task{task}-{job}"
    for d in ("node-results", "node-archives", "node-status"):
        (RUN / d).mkdir(exist_ok=True)
    work_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"mmo-{job}-{task}"
    work_root.mkdir(parents=True, exist_ok=True)
    stop = RUN / f"STOP-{route.kind}"
    threading.Thread(target=file_quota_guard, args=(stop,), daemon=True).start()
    try:
        record_versions(route, campaign, work_root)
    except Exception as e:
        print(f"version record failed: {e!r}", flush=True)
    queue = collections.deque((c, 1) for c in pending)
    done = collections.Counter()
    snapshot = RUN / "node-status" / f"{campaign}.json"

    def status():
        write_json(snapshot, {"campaign": campaign, "assigned": len(pending), "done": dict(done), "time": time.time(),
                              "host": socket.gethostname(), "paused": route.paused, "queue": len(queue)})

    def worker(index):
        time.sleep(index * route.start_gap)
        while not (stop.exists() or route.paused):
            with QUEUE_LOCK:
                item = queue.popleft() if queue else None
            if item is None:
                return
            case, attempt = item
            try:
                rec = run_attempt(case, route, work_root, attempt, campaign)
            except Exception as e:  # left not done; a later campaign reruns it from scratch
                print(f"case {case['case_id']} attempt {attempt} failed in the runner: {e!r}", flush=True)
                done["fault:runner"] += 1
                continue
            done[rec["status"] if rec["status"] == "complete" else f"fault:{rec['fault_kind']}"] += 1
            status()
            if rec["status"] != "complete" and attempt < MAX_ATTEMPTS and not route.paused:
                time.sleep(random.uniform(30, 90) * attempt)
                with QUEUE_LOCK:
                    queue.append((case, attempt + 1))

    status()
    time.sleep(30 * task)  # staggered start across the array (STOP-01)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(worker, i) for i in range(workers)]:
            f.result()
    status()
    print(json.dumps({"campaign": campaign, "assigned": len(pending), "done": done, "left": len(queue),
                      "paused": route.paused, "stopped": stop.exists()}), flush=True)
