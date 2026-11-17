#!/usr/bin/env python3
"""Great Lakes runner for the ISSP m-value reference run (29 October), copied from the ISSP live selection runner.

Changes from that runner, and only these: a case may name its research prompt (prompt_file; the two variants
change one sentence, the team's stance), and kind "reference" runs the analysis phase only (no commitment, no
scoring). Everything else, including the data writer, sandbox, CLI build, model, effort and relay, is unchanged.

Original description:

Reuses the validated pieces of the Codex runtime: the pinned Codex CLI 0.157
build, its bubblewrap guest layout, its configuration, the accepted billing
relay (which alone holds the API key) and its event parsing. One relay runs per
node, with its own ledger and dollar cap; a shared STOP file halts new requests
on every node.

Changes for this application: every data.csv (original and replays) is written by
write_countries (whole countries, rows copied verbatim in seeded random order, fixed
timestamp); the prompt's {{OUT_DIR}} is results/brief/run_01; the frozen program is
that folder's analysis_final.py, run with --data and --output.

Subcommands
  cases  --list cases.jsonl --task N --tasks K   run every K-th case from N
  score  --original-result R --datasets-dir D --out O   run an original's
         program on many datasets in a network-isolated sandbox
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
EXP = Path(os.environ.get("EXP_ROOT", "<cluster scratch>/issp-mvalue-reference-20261029"))
RUNTIME = Path("<cluster scratch>/luna-immigration-selection-20261015/runtime-0.157.0")
CLI = RUNTIME / "codex-x86_64-unknown-linux-musl"
CODE_MODE_HOST = RUNTIME / "codex-code-mode-host-x86_64-unknown-linux-musl"
CURATED_PYTHON = Path("<cluster scratch>/immigration-scientific-python-v1")
PYTHON = "python3.12"
PRICING = Path("<cluster scratch 2>/selectivereplay-20261016/fresh200-shared-accelerated1933-v2/control/pricing.json")
REAL_KEY = Path("<cluster scratch>/selectivereplay-native-cli-20261014/control/private/openai-key.txt")
# ChatGPT-subscription login ("other" Codex account), mounted read-only; never copied or archived.
SUB_AUTH = Path("<cluster scratch>/codex-private-journal-reported-pap-20261022/other-account/auth.json")
LIMIT_PATTERN = re.compile(r"usage limit|rate limit|too many requests|\b429\b|quota", re.I)
PROXY = "http://<campus proxy>:3128"
NO_PROXY = "localhost,127.0.0.1,::1"
MODEL = "gpt-6-luna"
EFFORT = "high"  # the accepted relay only forwards "high"; other values are for subscription-route tests
STOP = EXP / "STOP"
PAUSE = EXP / "PAUSE"  # manual hold: new sessions wait while it exists; it expires after 20 minutes

GUEST = {"workspace": "/workspace", "home": "/home/researcher", "tmp": "/tmp", "codex-home": "/codex-home"}
GUEST_CLI = "/opt/codex/codex"
GUEST_CMH = "/opt/codex/codex-code-mode-host"
GUEST_SCHEMA = "/run/scientific/schema.json"
GUEST_ENV = {
    "PATH": "/opt/codex:<cluster python 3.12.1>/bin:<cluster software>/arc/stacks/gcc/13.2.0/R/4.5.1/bin:"
            "<cluster software>/coe/o/image-libraries/220318/bin:<cluster software>/arc/gcc/13.2.0/bin:/usr/local/bin:/usr/bin",
    "LD_LIBRARY_PATH": "<cluster python 3.12.1>/lib:<cluster software>/arc/stacks/gcc/13.2.0/R/4.5.1/lib64/R/lib:"
                       "<cluster software>/coe/o/image-libraries/220318/lib:<cluster software>/arc/gcc/13.2.0/lib64",
    "LIBRARY_PATH": "<cluster software>/arc/gcc/13.2.0/lib64",
    "CPATH": "<cluster software>/arc/gcc/13.2.0/lib/gcc/x86_64-pc-linux-gnu/13.2.0/include-fixed:"
             "<cluster software>/arc/gcc/13.2.0/lib/gcc/x86_64-pc-linux-gnu/13.2.0/include",
    "PKG_CONFIG_PATH": "<cluster software>/arc/stacks/gcc/13.2.0/R/4.5.1/lib64/pkgconfig:"
                       "<cluster software>/coe/o/image-libraries/220318/lib/pkgconfig",
    "PYTHONPATH": "/opt/scientific-python", "PYTHONDONTWRITEBYTECODE": "1",
    "R_LIBS_SITE": "<cluster software>/arc/stacks/gcc/13.2.0/Rtidyverse/4.5.1/2025-09-30",
    "R_ENVIRON_USER": "/dev/null", "R_PROFILE_USER": "/dev/null", "R_ENVIRON": "/dev/null", "R_PROFILE": "/dev/null",
    "PYTHONNOUSERSITE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "HTTPS_PROXY": PROXY, "HTTP_PROXY": PROXY, "https_proxy": PROXY, "http_proxy": PROXY,
    "NO_PROXY": NO_PROXY, "no_proxy": NO_PROXY,
}
PUBLIC_ETC = ["/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/passwd", "/etc/group",
              "/etc/localtime", "/etc/ssl", "/etc/pki", "/etc/ld.so.cache", "/etc/fonts", "/etc/os-release",
              "/etc/services", "/etc/protocols", "/etc/alternatives"]
DISABLED_FEATURES = ("apps", "auth_elicitation", "browser_use", "browser_use_external",
    "browser_use_full_cdp_access", "computer_use", "fast_mode", "goals", "guardian_approval", "hooks",
    "image_generation", "in_app_browser", "in_app_chat", "in_app_dictation", "in_app_updates", "memories",
    "multi_agent", "multi_agent_v2", "personality", "plugin_sharing", "plugins", "remote_compaction_v2",
    "remote_plugin", "shell_snapshot", "skill_mcp_dependency_install", "skill_search",
    "tool_call_mcp_elicitation", "tool_suggest", "unbounded_connection_retries", "view_image",
    "workspace_dependencies")
DATA_SCOPE = ("Use the supplied packet as the sole research data source. Internet access may be used for "
              "technical documentation, software and package installation. Do not replace or augment research "
              "observations with downloaded datasets. Keep supplied packet files unchanged and write analysis "
              "artifacts separately.")
FIELDS = ("model", "null", "population", "outcome", "sample_restrictions")
COMMITMENT_SCHEMA = {"type": "object", "properties": {x: {"type": "string"} for x in FIELDS},
                     "required": list(FIELDS), "additionalProperties": False}
COMPARISON_SCHEMA = {"type": "object", "properties": {"model_null_match": {"type": "boolean"},
                     "reason": {"type": "string"}}, "required": ["model_null_match", "reason"],
                     "additionalProperties": False}
TIMEOUT = {"analysis": 3600, "commitment": 900, "comparison": 900}
OUT_REL = "results/brief/run_01"  # the same output folder in every session
SCORE_SCRIPT = OUT_REL + "/analysis_final.py"
FIXED_MTIME = 1792486800  # 2026-10-20 09:00 UTC: packet files carry the same timestamp in every workspace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)


# ---------------------------------------------------------------- file-light storage
# Scratch has a hard file-count quota shared with all earlier runs. Each node keeps
# one results file and one archive, appending cases as they finish, and replay
# datasets are rebuilt on the compute node from their seed instead of stored.

SINK_LOCK = threading.Lock()
CAMPAIGN = "local"
DONE = set()
FILE_QUOTA_STOP = 0.97


_SOURCE = {}
_SOURCE_LOCK = threading.Lock()


def source_rows(path):
    """Header and data lines of the source file, verbatim, grouped by iso_country (cached per process).
    The source has no quoted fields, so splitting on commas is exact."""
    with _SOURCE_LOCK:
        if path not in _SOURCE:
            with open(path) as f:
                header, *lines = f.read().splitlines()
            col = header.split(",").index("iso_country")
            by = {}
            for line in lines:
                by.setdefault(line.split(",")[col], []).append(line)
            _SOURCE[path] = (header, by)
        return _SOURCE[path]


def write_countries(spec, dest):
    """Write data.csv for any case: k whole countries (all of them when k is None), rows copied verbatim in seeded
    random order, the same trailing newline and modification time as every other packet."""
    header, by = source_rows(spec["source"])
    countries = sorted(by, key=int)
    if spec.get("k"):
        countries = random.Random(f"{spec['seed']}-bank-{spec['bank_index']}").sample(countries, spec["k"])
    rows = [line for c in countries for line in by[c]]
    random.Random(f"{spec['seed']}-order-{spec['bank_index']}").shuffle(rows)
    Path(dest).write_text("\n".join([header] + rows) + "\n")
    os.utime(dest, (FIXED_MTIME, FIXED_MTIME))


def load_results():
    """Finished cases by id, from per-case files (pilot) and per-node JSONL (later runs)."""
    out = {p.stem: json.loads(p.read_text()) for p in (EXP / "results").glob("*.json")}
    for p in sorted((EXP / "node-results").glob("*.results.jsonl")):
        for line in open(p):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out[rec["case_id"]] = rec
    return out


def load_done():
    return set(load_results())


def sink(rec, case_dir, limited):
    """Append the case's result and archive to this node's single files."""
    packed = Path(case_dir).parent / f"{rec['case_id']}-a{rec.get('attempt', 1)}.tar.gz"
    with tarfile.open(packed, "w:gz") as t:
        # data.csv is rebuilt from its spec, so it is not archived.
        t.add(case_dir, arcname=rec["case_id"], filter=lambda ti: None if ti.name.endswith("auth.json")
              or ti.name.endswith("/packet/data.csv") else ti)
    kind = "limited" if limited else "results"
    with SINK_LOCK:
        with open(EXP / "node-results" / f"{CAMPAIGN}.{kind}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with tarfile.open(EXP / "node-archives" / f"{CAMPAIGN}.tar", "a") as t:
            t.add(packed, arcname=packed.name)
    packed.unlink()


def file_quota_fraction():
    out = subprocess.run(["scratch-quota", "<allocation>"], capture_output=True, text=True, timeout=120).stdout
    for line in out.splitlines():
        if "FILESET" in line and "|" in line:
            nums = [x for x in line.split("|")[1].split() if x.isdigit()]
            if len(nums) >= 3 and int(nums[2]) > 0:
                return int(nums[0]) / int(nums[2])
    return None


def file_quota_guard():
    while True:
        try:
            frac = file_quota_fraction()
            if frac is not None and frac >= FILE_QUOTA_STOP and not STOP.exists():
                STOP.write_text(json.dumps({"reason": "scratch file-count quota", "fraction": frac,
                                            "time": time.time()}))
        except Exception:
            pass
        time.sleep(300)


# ---------------------------------------------------------------- subscription pacing
# The subscription route has an unpublished request-rate limit (HTTP 429). Each node
# adds one live session per completed session and halves its sessions after a 429.
# A throttled case is retried after a jittered backoff: a 429 is not an agent outcome.

MAX_TRIES = 6
GATE = None


class Gate:
    def __init__(self, start, cap, gap):
        self.limit, self.cap, self.gap = start, cap, gap
        self.active, self.last_start, self.last_cut = 0, 0.0, 0.0
        self.cv = threading.Condition()

    def acquire(self):
        with self.cv:
            while not STOP.exists():
                wait = self.last_start + self.gap - time.time()
                if self.active < self.limit and wait <= 0:
                    self.active += 1
                    self.last_start = time.time()
                    return True
                self.cv.wait(timeout=wait if self.active < self.limit else 30)
            return False

    def release(self, limited):
        with self.cv:
            self.active -= 1
            if not limited:
                self.limit = min(self.cap, self.limit + 1)
            elif time.time() - self.last_cut > 120:
                self.limit = max(1, self.limit // 2)
                self.last_cut = time.time()
            self.cv.notify_all()


# ---------------------------------------------------------------- sandbox

def config_text(port=None):
    """port=None selects the ChatGPT-subscription route (Codex's own login, no relay)."""
    v = {"model": MODEL, "model_reasoning_effort": EFFORT,
         "service_tier": "default", "approval_policy": "never", "sandbox_mode": "danger-full-access",
         "agents.enabled": False, "web_search": "disabled", "project_doc_max_bytes": 0,
         "project_doc_fallback_filenames": [], "shell_environment_policy.inherit": "none",
         "shell_environment_policy.experimental_use_profile": False,
         "shell_environment_policy.ignore_default_excludes": False,
         "memories.generate_memories": False, "memories.use_memories": False,
         "developer_instructions": DATA_SCOPE}
    if port is None:
        v.update({"model_provider": "openai_https", "forced_login_method": "chatgpt",
                  "cli_auth_credentials_store": "file",
                  "model_providers.openai_https.name": "OpenAI",
                  "model_providers.openai_https.wire_api": "responses",
                  "model_providers.openai_https.requires_openai_auth": True,
                  "model_providers.openai_https.supports_websockets": False,
                  "model_providers.openai_https.request_max_retries": 2,
                  "model_providers.openai_https.stream_max_retries": 2})
    else:
        v.update({"model_provider": "scientific_relay",
                  "model_providers.scientific_relay.name": "Frozen scientific billing relay",
                  "model_providers.scientific_relay.base_url": f"http://127.0.0.1:{port}/v1",
                  "model_providers.scientific_relay.wire_api": "responses",
                  "model_providers.scientific_relay.env_key": "CODEX_RELAY_TOKEN",
                  "model_providers.scientific_relay.requires_openai_auth": False,
                  "model_providers.scientific_relay.supports_websockets": False,
                  "model_providers.scientific_relay.request_max_retries": 2,
                  "model_providers.scientific_relay.stream_max_retries": 2,
                  "model_providers.scientific_relay.http_headers.X-Scientific-Service-Tier": "default"})
    for k, val in GUEST_ENV.items():
        v[f"shell_environment_policy.set.{k}"] = val
    for k, val in (("HOME", GUEST["home"]), ("TMPDIR", GUEST["tmp"])):
        v[f"shell_environment_policy.set.{k}"] = val
    v.update({f"features.{n}": False for n in DISABLED_FEATURES})
    v.update({"features.code_mode_host": True, "features.shell_tool": True, "features.unified_exec": True})
    return "\n".join(f"{k} = {json.dumps(x)}" for k, x in sorted(v.items())) + "\n"


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


def bwrap(case_dir, guest_cmd, *, token=None, schema=None, network=True, ro_workspace=False,
          extra_ro=None, extra_rw=None, cwd=GUEST["workspace"]):
    c = Path(case_dir)
    args = ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "/usr/bin/bwrap", "--die-with-parent",
            "--unshare-user", "--unshare-pid", "--unshare-ipc", "--new-session"]
    if not network:
        args.append("--unshare-net")
    args += ["--tmpfs", "/", "--dev", "/dev", "--proc", "/proc"] + public_mounts()
    for name, guest in GUEST.items():
        host = c / name
        if name == "workspace" and ro_workspace:
            args += ["--ro-bind", str(host), guest]
        else:
            args += ["--bind", str(host), guest]
    args += ["--dir", "/opt/codex", "--ro-bind", str(CLI), GUEST_CLI, "--ro-bind", str(CODE_MODE_HOST), GUEST_CMH]
    if (c / "packet").is_dir():
        for f in sorted((c / "packet").iterdir()):
            args += ["--ro-bind", str(f), f"{GUEST['workspace']}/{f.name}"]
    if (c / "config.toml").exists():
        args += ["--ro-bind", str(c / "config.toml"), GUEST["codex-home"] + "/config.toml"]
    if (c / "route-subscription").exists():
        args += ["--ro-bind", str(SUB_AUTH), GUEST["codex-home"] + "/auth.json"]
    if schema is not None:
        args += ["--dir", "/run/scientific", "--ro-bind", str(schema), GUEST_SCHEMA]
    for guest, host in (extra_ro or {}).items():
        args += ["--dir", str(Path(guest).parent), "--ro-bind", str(host), guest]
    for guest, host in (extra_rw or {}).items():
        args += ["--dir", guest, "--bind", str(host), guest]
    env = dict(GUEST_ENV, HOME=GUEST["home"], CODEX_HOME=GUEST["codex-home"], TMPDIR=GUEST["tmp"])
    if token is not None:
        env["CODEX_RELAY_TOKEN"] = token
    for k, val in sorted(env.items()):
        args += ["--setenv", k, val]
    return args + ["--chdir", cwd, "--"] + list(guest_cmd)


def cli_command(session_id=None, schema=False):
    cmd = [GUEST_CLI, "exec", "--cd", GUEST["workspace"], "--strict-config", "--ignore-rules",
           "--model", MODEL, "--skip-git-repo-check"]
    if session_id:
        cmd += ["resume", session_id]
    cmd += ["--json"]
    if schema:
        cmd += ["--output-schema", GUEST_SCHEMA]
    return cmd + ["-"]


def parse_events(text, expected=None):
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    sessions = {e["thread_id"] for e in events if e.get("type") == "thread.started"}
    if len(sessions) != 1:
        raise ValueError(f"expected one session, found {len(sessions)}")
    session = next(iter(sessions))
    if expected and session != expected:
        raise ValueError("resume changed session")
    life = [e.get("type") for e in events if e.get("type") in ("turn.started", "turn.completed", "turn.failed")]
    if not life or life[-1] != "turn.completed":
        raise ValueError("turn did not complete")
    answers = [e["item"]["text"] for e in events if e.get("type") == "item.completed"
               and e.get("item", {}).get("type") == "agent_message"]
    if not answers:
        raise ValueError("no final agent message")
    usage = [e.get("usage") for e in events if e.get("type") == "turn.completed" and e.get("usage")]
    return {"session_id": session, "answer": answers[-1], "usage": usage[-1] if usage else None,
            "n_events": len(events),
            "n_commands": sum(1 for e in events if e.get("type") == "item.completed"
                              and e.get("item", {}).get("type") in ("command_execution", "function_call",
                                                                     "custom_tool_call", "local_shell_call"))}


def run_phase(case_dir, phase, prompt, token, *, session_id=None, schema=None):
    c = Path(case_dir)
    log = c / "logs"
    log.mkdir(exist_ok=True)
    schema_file = None
    if schema is not None:
        schema_file = c / f"{phase}-schema.json"
        schema_file.write_text(json.dumps(schema))
    cmd = bwrap(c, cli_command(session_id, schema is not None), token=token, schema=schema_file)
    t0 = time.time()
    timed_out = False
    with open(log / f"{phase}.events.jsonl", "w") as out, open(log / f"{phase}.stderr", "w") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True)
        try:
            proc.communicate(prompt, timeout=TIMEOUT[phase])
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
    rec = {"phase": phase, "seconds": round(time.time() - t0, 1), "returncode": proc.returncode,
           "timed_out": timed_out}
    try:
        rec.update(parse_events((log / f"{phase}.events.jsonl").read_text(), session_id))
        rec["ok"] = not timed_out and proc.returncode == 0
    except ValueError as e:
        rec.update(ok=False, error=str(e))
        errors = [l for l in (log / f"{phase}.events.jsonl").read_text().splitlines()
                  if '"error"' in l or '"turn.failed"' in l] + [(log / f"{phase}.stderr").read_text()[-4000:]]
        if any(LIMIT_PATTERN.search(x) for x in errors):
            rec["limit_error"] = True
    return rec


def parse_json_answer(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


# ---------------------------------------------------------------- scoring

def score_dataset(workspace, data_path, out_dir):
    """Run the agent's analysis_final.py on one dataset with no network and a scratch copy of its workspace."""
    run = Path(out_dir)
    run.mkdir(parents=True, exist_ok=True)
    for name in GUEST:
        (run / name).mkdir(exist_ok=True)
    for f in Path(workspace).iterdir():
        if f.name == "data.csv":
            continue
        if f.is_file() and f.stat().st_size < 20_000_000:
            shutil.copy2(f, run / "workspace" / f.name)
        elif f.is_dir() and f.name not in ("__pycache__",):
            shutil.copytree(f, run / "workspace" / f.name, symlinks=True,
                            ignore=shutil.ignore_patterns("*.csv", "*.parquet", "*.rds", "*.RData"))
    (run / "scoreout").mkdir(exist_ok=True)
    cmd = bwrap(run, ["python3", SCORE_SCRIPT, "--data", "/scoredata/data.csv", "--output", "/scoreout/out.json"],
                network=False, extra_ro={"/scoredata/data.csv": data_path}, extra_rw={"/scoreout": run / "scoreout"})
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        out = run / "scoreout" / "out.json"
        res = json.loads(out.read_text()) if out.exists() else None
        return {"returncode": p.returncode, "result": res, "stderr": p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "result": None, "stderr": "timeout"}


def p_from(result):
    try:
        p = float(result["p_value"])
        return p if 0 <= p <= 1 else None
    except (TypeError, KeyError, ValueError):
        return None


# ---------------------------------------------------------------- relay

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_relay(node_dir, cap, campaign):
    node_dir = Path(node_dir)
    (node_dir / "receipts").mkdir(parents=True, exist_ok=True)
    dummy = node_dir / "dummy-token"
    dummy.write_text(secrets.token_hex(32) + "\n")
    dummy.chmod(0o600)
    alloc = {"cap_usd": cap, "campaign": campaign, "allocation_sha256": hashlib.sha256(campaign.encode()).hexdigest()}
    (node_dir / "allocation.json").write_text(json.dumps(alloc))
    port = free_port()
    env = dict(os.environ, HTTPS_PROXY=PROXY, HTTP_PROXY=PROXY, https_proxy=PROXY, http_proxy=PROXY,
               NO_PROXY=NO_PROXY, no_proxy=NO_PROXY)
    cmd = [PYTHON, str(HERE / "relay" / "relay_entrypoint.py"),
           "--accepted-relay", str(HERE / "relay" / "accepted_relay.py"),
           "--connect-adapter", str(HERE / "relay" / "linux_openai_connect.py"),
           "--ledger", str(node_dir / "ledger.sqlite"), "--receipts", str(node_dir / "receipts"),
           "--pricing", str(PRICING), "--allocation-metadata", str(node_dir / "allocation.json"),
           "--cap-usd", str(cap), "--real-key-file", str(REAL_KEY), "--dummy-token-file", str(dummy),
           "--port", str(port), "--max-physical-upstream-requests", "32",
           "--stop-new-claims-file", str(node_dir / "fault-stop")]
    proc = subprocess.Popen(cmd, env=env, stdout=open(node_dir / "relay.out", "w"),
                            stderr=open(node_dir / "relay.err", "w"))
    for _ in range(100):
        if (node_dir / "ledger.sqlite").exists():
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                return proc, port
            except OSError:
                pass
        if proc.poll() is not None:
            raise RuntimeError("relay exited: " + (node_dir / "relay.err").read_text()[-2000:])
        time.sleep(0.3)
    raise RuntimeError("relay did not start")


def ledger_spend(ledger):
    import sqlite3
    db = sqlite3.connect(str(ledger))
    try:
        charged = db.execute("SELECT COALESCE(SUM(charged),0) FROM requests").fetchone()[0]
        est = db.execute("SELECT COALESCE(SUM(cost_estimate),0) FROM requests").fetchone()[0]
        n = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        return {"charged_upper_usd": charged, "estimate_usd": est, "requests": n}
    finally:
        db.close()


# ---------------------------------------------------------------- cases

def usage_guard(case_dir, rec):
    """Record the subscription's limit usage from this session's log; stop new sessions before
    the weekly allowance runs out or paid credits start being drawn."""
    last = None
    for p in (Path(case_dir) / "codex-home" / "sessions").rglob("*.jsonl"):
        for line in open(p, errors="replace"):
            if '"rate_limits"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            rl = d.get("payload", {}).get("rate_limits")
            if rl and (last is None or d.get("timestamp", "") > last[0]):
                last = (d.get("timestamp", ""), rl)
    if last is None:
        return
    rl = last[1]
    weekly = (rl.get("primary") or {}).get("used_percent")
    credits = (rl.get("credits") or {}).get("balance")
    rec["rate_limits_end"] = {"time": last[0], "weekly_used_percent": weekly, "credits_balance": credits}
    baseline = EXP / "credits_baseline.txt"
    try:
        if credits is not None and not baseline.exists():
            baseline.write_text(str(credits))
        drawn = credits is not None and float(credits) < float(baseline.read_text()) - 1e-6
    except (OSError, ValueError):
        drawn = False
    if (weekly is not None and float(weekly) >= 85) or drawn:
        STOP.write_text(json.dumps({"reason": "subscription usage guard", "weekly_used_percent": weekly,
                                    "credits_balance": credits, "case": rec["case_id"], "time": time.time()}))


def research_prompt(case):
    name = case.get("prompt_file", "research_prompt.txt")
    return (HERE / name).read_text().replace("{{OUT_DIR}}", f"{GUEST['workspace']}/{OUT_REL}")


def prepare_case(case, c, port, subscription=False):
    """Build a case's directories, packet and configuration. Only data.csv differs between the original and a
    replay (check_workspaces.py verifies this before launch)."""
    for name in list(GUEST) + ["packet"]:
        (c / name).mkdir(parents=True, exist_ok=True)
    (c / "workspace" / OUT_REL).mkdir(parents=True, exist_ok=True)
    write_countries(case["countries"], c / "packet" / "data.csv")
    shutil.copy2(HERE / "CODEBOOK.md", c / "packet" / "CODEBOOK.md")
    os.utime(c / "packet" / "CODEBOOK.md", (FIXED_MTIME, FIXED_MTIME))
    (c / "config.toml").write_text(config_text(port))
    if subscription:
        (c / "route-subscription").touch()


def run_case(case, port, admission, work_root):
    for attempt in range(1, MAX_TRIES + 1):
        rec = run_attempt(case, port, admission, work_root, attempt)
        if not rec.get("limited"):
            return rec
        time.sleep(random.uniform(60, 240) * attempt)
    return rec


def run_attempt(case, port, admission, work_root, attempt):
    cid = case["case_id"]
    if cid in DONE:
        return {"case_id": cid, "status": "already_done"}
    c = Path(work_root) / cid
    prepare_case(case, c, port, subscription=admission is None)
    token = secrets.token_hex(32) if admission is not None else None
    # Admissions are single-use per id and the relay requires 32 hex characters, so retries get a derived id.
    admit_id = cid if attempt == 1 else hashlib.sha256(f"{cid}-a{attempt}".encode()).hexdigest()[:32]
    fault = Path(admission.path).parent / "fault-stop" if admission is not None else None
    while True:
        if STOP.exists():
            return {"case_id": cid, "status": "stopped_before_admission"}
        if PAUSE.exists():
            # Subscription usage limit hit somewhere: hold new sessions for 20 minutes.
            if time.time() - PAUSE.stat().st_mtime > 1200:
                try:
                    shutil.move(str(PAUSE), EXP / "faults" / f"pause-{int(time.time())}.txt")
                except FileNotFoundError:
                    pass
            time.sleep(60)
            continue
        if admission is None:
            break
        if fault.exists():
            # A provider fault on this node: pause new admissions, log it, then resume.
            time.sleep(60)
            try:
                dest = EXP / "faults" / f"{Path(admission.path).parent.name}-{int(time.time())}.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fault), dest)
            except FileNotFoundError:
                pass
            continue
        info = admission.try_admit(admit_id, token)
        if info["admitted"]:
            break
        time.sleep(15)
    if admission is None and not GATE.acquire():
        return {"case_id": cid, "status": "stopped_before_admission"}
    rec = {"case_id": cid, "kind": case["kind"], "meta": case.get("meta", {}), "started": time.time(),
           "route": "subscription" if admission is None else "api", "effort": EFFORT, "attempt": attempt,
           "data_sha256": sha(c / "packet" / "data.csv"), "phases": {}}
    try:
        a = run_phase(c, "analysis", research_prompt(case), token)
        rec["phases"]["analysis"] = a
        if a.get("ok"):
            rec["final_report"] = a["answer"][:20000]
        if a.get("ok") and case["kind"] != "reference":
            cm = run_phase(c, "commitment", (HERE / "commitment_prompt.txt").read_text(), token,
                           session_id=a["session_id"], schema=COMMITMENT_SCHEMA)
            rec["phases"]["commitment"] = cm
            if cm.get("ok"):
                try:
                    rec["commitment"] = parse_json_answer(cm["answer"])
                except (ValueError, json.JSONDecodeError) as e:
                    rec["commitment_error"] = str(e)
            if case["kind"] == "replay" and rec.get("commitment"):
                prompt = (HERE / "comparison_prompt.txt").read_text().replace(
                    "{{TARGET_JSON}}", json.dumps(case["target_commitment"], sort_keys=True))
                cp = run_phase(c, "comparison", prompt, token, session_id=a["session_id"],
                               schema=COMPARISON_SCHEMA)
                rec["phases"]["comparison"] = cp
                if cp.get("ok"):
                    try:
                        rec["comparison"] = parse_json_answer(cp["answer"])
                    except (ValueError, json.JSONDecodeError) as e:
                        rec["comparison_error"] = str(e)
        has_score = (c / "workspace" / SCORE_SCRIPT).exists()
        rec["has_score_py"] = has_score
        out = c / "workspace" / OUT_REL
        if out.is_dir():
            rec["files"] = sorted(p.name for p in out.iterdir())
            for name in ("results_final.json", "BRIEF.md", "ANALYSIS_PATHS.jsonl"):
                if (out / name).is_file():
                    rec[name] = (out / name).read_text(errors="replace")[:300000]
        if case["kind"] == "original" and has_score:
            # Preserve the frozen deliverable workspace for bank scoring.
            frozen = EXP / "originals" / cid / "workspace"
            if frozen.exists():
                shutil.rmtree(frozen)
            shutil.copytree(c / "workspace", frozen, symlinks=True,
                            ignore=shutil.ignore_patterns("data.csv", "CODEBOOK.md"))
            s = score_dataset(frozen, str(c / "packet" / "data.csv"), c / "score-original")
            rec["original_score"] = s
            rec["u"] = p_from(s["result"])
    except Exception as e:  # recorded, never retried automatically
        rec["runner_error"] = f"{type(e).__name__}: {e}"
    finally:
        if admission is not None:
            try:
                admission.close(admit_id)
            except Exception as e:
                rec["close_error"] = str(e)
        rec["finished"] = time.time()
        if admission is None:
            try:
                usage_guard(c, rec)
            except Exception as e:
                rec["usage_guard_error"] = str(e)
        limited = any(p.get("limit_error") for p in rec["phases"].values())
        rec["limited"] = limited
        if admission is None:
            GATE.release(limited)
        sink(rec, c, limited)
        shutil.rmtree(c, ignore_errors=True)
    return rec


FROZEN_FILES = ("research_prompt.txt", "research_prompt_anti.txt", "research_prompt_nonpartisan.txt", "commitment_prompt.txt", "comparison_prompt.txt", "CODEBOOK.md",
                "runner.py", "make_replays.py", "relay/accepted_relay.py",
                "relay/case_budget.py", "relay/relay_entrypoint.py", "relay/linux_openai_connect.py")


def record_versions(node_dir, campaign):
    """CLI version from inside the sandbox plus hashes of the frozen files, once per node."""
    run = Path(node_dir) / "version-check"
    for name in GUEST:
        (run / name).mkdir(parents=True, exist_ok=True)
    p = subprocess.run(bwrap(run, [GUEST_CLI, "--version"], network=False), capture_output=True, text=True,
                       timeout=120)
    rec = {"campaign": campaign, "host": socket.gethostname(), "time": time.time(), "model": MODEL,
           "effort": EFFORT, "cli_version": (p.stdout + p.stderr).strip()[:200], "cli_sha256": sha(CLI),
           "files": {f: sha(HERE / f) for f in FROZEN_FILES if (HERE / f).is_file()}}
    write_json(EXP / "node-status" / f"{campaign}.versions.json", rec)


def cmd_cases(a):
    global EFFORT, CAMPAIGN, DONE, GATE
    EFFORT = a.effort
    cases = [json.loads(l) for l in open(a.list) if l.strip()]
    mine = [c for i, c in enumerate(cases) if i % a.tasks == a.task]
    job = os.environ.get("SLURM_JOB_ID", "local")
    node_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"node-{job}-{a.task}"
    node_dir.mkdir(parents=True, exist_ok=True)
    campaign = f"{a.campaign}-task{a.task}"
    CAMPAIGN = f"{campaign}-{job}"
    for d in ("node-results", "node-archives", "faults", "node-status"):
        (EXP / d).mkdir(exist_ok=True)
    DONE = load_done()
    threading.Thread(target=file_quota_guard, daemon=True).start()
    try:
        record_versions(node_dir, campaign)
    except Exception as e:
        print(f"version record failed: {e!r}", file=sys.stderr, flush=True)
    relay, port, admission = None, None, None
    if a.route == "api":
        relay, port = start_relay(node_dir, a.cap, campaign)
        sys.path.insert(0, str(HERE / "relay"))
        import case_budget
        admission = case_budget.Admission(node_dir / "ledger.sqlite", a.cap,
                                          working_headroom_per_active_case_usd=0.3)
    elif not SUB_AUTH.is_file():
        raise SystemExit("subscription login missing")
    else:
        GATE = Gate(min(a.start_concurrency, a.concurrency), a.concurrency, a.start_gap)
    work_root = node_dir / "cases"
    status_path = EXP / "node-status" / f"{campaign}.json"

    def snapshot(done):
        if a.route == "api":
            try:
                spend = ledger_spend(node_dir / "ledger.sqlite")
            except Exception as e:
                spend = {"error": str(e)}
        else:
            spend = {"route": "subscription", "gate_limit": GATE.limit, "gate_active": GATE.active}
        write_json(status_path, {"campaign": campaign, "assigned": len(mine), "done": done,
                                 "spend": spend, "time": time.time(), "host": socket.gethostname()})

    done = 0
    try:
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            futs = [pool.submit(run_case, c, port, admission, work_root) for c in mine]
            for f in cf.as_completed(futs):
                done += 1
                if f.exception() is not None:
                    print(f"case thread failed: {f.exception()!r}", file=sys.stderr, flush=True)
                snapshot(done)
    finally:
        snapshot(done)
        if relay is not None:
            close_relay(relay, node_dir, campaign)


def close_relay(relay, node_dir, campaign):
    relay.terminate()
    try:
        relay.wait(timeout=30)
    except subprocess.TimeoutExpired:
        relay.kill()
    dest = EXP / "ledgers" / campaign
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("ledger.sqlite", "allocation.json", "relay.err"):
        if (node_dir / name).exists():
            shutil.copy2(node_dir / name, dest / name)
    with tarfile.open(dest / "receipts.tar.gz", "w:gz") as t:
        t.add(node_dir / "receipts", arcname="receipts")


def cmd_score(a):
    """Score every dataset in a directory with one original's frozen program."""
    original = json.loads(Path(a.original_result).read_text())
    workspace = EXP / "originals" / original["case_id"] / "workspace"
    datasets = sorted(Path(a.datasets_dir).glob("*.csv"))
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"score-{original['case_id']}"

    def one(d):
        r = score_dataset(workspace, str(d), tmp / d.stem)
        shutil.rmtree(tmp / d.stem, ignore_errors=True)
        return d.stem, p_from(r["result"]), (r["stderr"][-300:] if p_from(r["result"]) is None else "")

    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        rows = list(pool.map(one, datasets))
    write_json(a.out, {"case_id": original["case_id"], "u": original.get("u"),
                       "scores": {k: p for k, p, _ in rows}, "errors": {k: e for k, p, e in rows if p is None}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--list", required=True)
    c.add_argument("--task", type=int, required=True)
    c.add_argument("--tasks", type=int, required=True)
    c.add_argument("--cap", type=float, required=True)
    c.add_argument("--campaign", required=True)
    c.add_argument("--concurrency", type=int, default=32)
    c.add_argument("--route", choices=["api", "subscription"], default="api")
    c.add_argument("--effort", default="high")
    c.add_argument("--start-concurrency", type=int, default=8, help="subscription route: initial live sessions")
    c.add_argument("--start-gap", type=float, default=5.0, help="subscription route: seconds between session starts")
    s = sub.add_parser("score")
    s.add_argument("--original-result", required=True)
    s.add_argument("--datasets-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    {"cases": cmd_cases, "score": cmd_score}[args.cmd](args)
