#!/usr/bin/env python3
"""Persona pilot reproducing the agentic-bootstrap design of Miao et al. (2026) with gpt-6-luna.

Mirrors codes/agentic_bootstrap.mjs at commit 7c8925f: one autonomous agent session per run,
the persona text as the system-level (developer) instruction, and the assembled task prompt as
the only user message. The task prompt follows the paper's Appendix A: the ISSP data prompt with
the paper's sign convention, ten rounds of ten models across seven axes, each round opening with
"Recall your perspective on this research question.", and a final top-10 and headline selection.
As in their harness, the output directory is results/<persona>/run_<id> and the agent's commands
have no internet access. The sandbox, pinned Codex CLI and ChatGPT-subscription login come from
the open-ended simulation runner (runner.py, copied unchanged).

Subcommands
  make-cases --runs R --out cases.jsonl [--delivery developer|system]   R runs per persona, interleaved
  run --list cases.jsonl [--concurrency N] [--results results.jsonl]    run every case on this node
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import secrets
import shutil
import sys
import tarfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = Path(os.environ["EXP_ROOT"])
sys.path.insert(0, str(HERE))
import runner  # noqa: E402  (reads EXP_ROOT at import, so its STOP file is this experiment's)

PROMPTS = HERE / "prompts"
DATA = EXP / "data" / "data_clean.csv"
GUEST_DATA = runner.GUEST["workspace"] + "/data_clean.csv"
SYSTEM_PROMPT_GUEST = runner.GUEST["codex-home"] + "/persona_system_prompt.md"
QUESTION = "How does immigration affect public support for social welfare programs?"
NUM_ROUNDS = 10
PERSONAS = ("anti_immigration", "pro_immigration", "neutral")
NO_INTERNET = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "NO_PROXY", "no_proxy")
TIMEOUT_SECONDS = 4 * 3600
TURN_TIMEOUT_SECONDS = 3600
MAX_TRIES = 3
LOCK = threading.Lock()
RESULTS = EXP / "results.jsonl"


def read(name):
    return (PROMPTS / name).read_text().rstrip()


def round_instructions(out_dir):
    """expandRoundTemplate from prompts/load_prompts.mjs."""
    intro, rest = read("_round_template.md").split("%%REPEAT_FOR_R_FROM_2_TO_NUM_ROUNDS%%")
    body, outro = rest.split("%%END_REPEAT%%")
    blocks = [body.replace("{{R}}", str(r)).replace("{{R_MINUS_1}}", str(r - 1)).replace("{{OUT_DIR}}", out_dir)
              for r in range(2, NUM_ROUNDS + 1)]
    return intro.replace("{{OUT_DIR}}", out_dir) + "".join(blocks) + outro


def task_prompt(out_dir, instructions="10round_7axis_paper.md", data_prompt="issp_paper.md"):
    """loadTaskPrompt from prompts/load_prompts.mjs, with the paper's data and instruction text."""
    data = read(data_prompt).replace("{{DATA_PATH}}", GUEST_DATA).replace("{{RESEARCH_QUESTION}}", QUESTION)
    instructions = (read(instructions).replace("{{NUM_ROUNDS}}", str(NUM_ROUNDS))
                    .replace("{{OUT_DIR}}", out_dir).replace("{{ROUND_INSTRUCTIONS}}", round_instructions(out_dir)))
    return f"{data}\n\n{instructions}\n"


def turn_prompts(out_dir, instructions="10round_7axis_paper.md", data_prompt="issp_paper.md"):
    """The text of task_prompt split into user turns: the data prompt, instructions and Round 1
    first, then one turn per later round, then the final step. Used when an agent would otherwise
    write every round up front instead of revising after seeing each round's results."""
    data = read(data_prompt).replace("{{DATA_PATH}}", GUEST_DATA).replace("{{RESEARCH_QUESTION}}", QUESTION)
    head, tail = (read(instructions).replace("{{NUM_ROUNDS}}", str(NUM_ROUNDS))
                  .replace("{{OUT_DIR}}", out_dir).split("{{ROUND_INSTRUCTIONS}}"))
    intro, rest = read("_round_template.md").split("%%REPEAT_FOR_R_FROM_2_TO_NUM_ROUNDS%%")
    body = rest.split("%%END_REPEAT%%")[0]
    turns = [f"{data}\n\n{head}{intro.replace('{{OUT_DIR}}', out_dir)}".rstrip().removesuffix("---").rstrip()]
    turns += [body.replace("{{R}}", str(r)).replace("{{R_MINUS_1}}", str(r - 1)).replace("{{OUT_DIR}}", out_dir)
              .strip().removesuffix("---").rstrip() for r in range(2, NUM_ROUNDS + 1)]
    return turns + [tail.strip()]


def config_for(persona, delivery="developer"):
    """The runner's subscription-route config with no proxy in the agent's shell environment
    (compute nodes reach the internet only via the proxy). delivery="developer" adds the persona as
    developer instructions on top of Codex's built-in prompt; delivery="system" replaces the built-in
    prompt with the persona, as Miao et al.'s systemPrompt replaces Claude Code's."""
    lines = [l for l in runner.config_text(None).splitlines()
             if not l.startswith("developer_instructions ")
             and not any(l.startswith(f"shell_environment_policy.set.{k} ") for k in NO_INTERNET)]
    if delivery == "original":
        return runner.config_text(None)  # the open-ended application's own configuration, internet included
    if delivery == "system":
        lines.append(f"model_instructions_file = {json.dumps(SYSTEM_PROMPT_GUEST)}")
    else:
        lines.append(f"developer_instructions = {json.dumps(read('persona/' + persona + '.md'))}")
    return "\n".join(sorted(lines)) + "\n"


def read_csv_rows(path):
    import csv
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def session_settings(case_dir):
    """Model and effort actually used, from the CLI's own session log inside the sandbox."""
    models, efforts = set(), set()
    for p in (Path(case_dir) / "codex-home" / "sessions").rglob("*.jsonl"):
        for line in open(p, errors="replace"):
            if '"turn_context"' not in line:
                continue
            try:
                payload = json.loads(line).get("payload", {})
            except ValueError:
                continue
            models.add(payload.get("model"))
            efforts.add(payload.get("effort") or payload.get("reasoning_effort"))
    return sorted(m for m in models if m), sorted(e for e in efforts if e)


def archive(case_dir, case_id, attempt):
    skip = ("/packet", "/workspace/data_clean.csv", "/workspace/data.csv", "/workspace/CODEBOOK.md", "/codex-home/auth.json")
    dest = EXP / "archives" / f"{case_id}-a{attempt}.tar.gz"
    with tarfile.open(dest, "w:gz") as t:
        t.add(case_dir, arcname=case_id,
              filter=lambda ti: None if any(ti.name.endswith(s) or (s + "/") in ti.name + "/" for s in skip) else ti)
    return str(dest)


def run_attempt(case, attempt, work_root):
    cid = case["case_id"]
    c = work_root / f"{cid}-a{attempt}"
    for name in list(runner.GUEST) + ["packet"]:
        (c / name).mkdir(parents=True, exist_ok=True)
    packet = case.get("packet_files", {"data_clean.csv": case.get("data_file", "data_clean.csv")})
    for name, src in packet.items():
        shutil.copy2(EXP / "data" / src, c / "packet" / name)
    out_rel = f"results/{case.get('out_name', case['persona'])}/run_{case['run']:02d}"
    (c / "workspace" / out_rel).mkdir(parents=True)
    delivery = case.get("delivery", "developer")
    (c / "config.toml").write_text(config_for(case["persona"], delivery))
    if delivery == "system":
        (c / "codex-home" / Path(SYSTEM_PROMPT_GUEST).name).write_text(read("persona/" + case["persona"] + ".md") + "\n")
    (c / "route-subscription").touch()
    turns = case.get("turns", "single")
    rec = {"case_id": cid, "persona": case["persona"], "run": case["run"], "delivery": delivery,
           "arm": case.get("arm"), "stance": case.get("stance"), "turns": turns, "attempt": attempt, "started": time.time(), "out_dir": "/workspace/" + out_rel}
    try:
        if turns == "per_round":
            phases, sid = [], None
            for k, prompt in enumerate(turn_prompts(rec["out_dir"], case.get("instructions", "10round_7axis_paper.md"),
                                                    case.get("data_prompt", "issp_paper.md"))):
                name = f"r{k + 1}" if k < NUM_ROUNDS else "final"
                runner.TIMEOUT[name] = TURN_TIMEOUT_SECONDS
                p = runner.run_phase(c, name, prompt, None, session_id=sid)
                phases.append(dict(p, answer=p.get("answer", "")[:4000]))
                if not p.get("ok"):
                    break
                sid = p["session_id"]
            rec["phases"] = phases
            rec["analysis"] = {"ok": len(phases) == NUM_ROUNDS + 1 and phases[-1].get("ok", False),
                               "seconds": sum(p.get("seconds", 0) for p in phases), "turns_completed": len(phases),
                               "limit_error": any(p.get("limit_error") for p in phases),
                               "error": next((p.get("error") for p in phases if p.get("error")), None)}
        elif case.get("prompt_file"):  # a complete prompt used verbatim, with only the output folder filled in
            rec["analysis"] = runner.run_phase(c, "analysis", read(case["prompt_file"]).replace("{{OUT_DIR}}", rec["out_dir"]), None)
        else:
            rec["analysis"] = runner.run_phase(c, "analysis", task_prompt(rec["out_dir"], case.get("instructions", "10round_7axis_paper.md"),
                                                                          case.get("data_prompt", "issp_paper.md")), None)
        rec["models_used"], rec["efforts_used"] = session_settings(c)
        out = c / "workspace" / out_rel
        rec["files"] = sorted(p.name for p in out.iterdir())
        rec["rounds"] = {r: read_csv_rows(out / f"results_r{r}.csv") for r in range(1, NUM_ROUNDS + 1)
                         if (out / f"results_r{r}.csv").exists()}
        for name in ("results_top10", "results_final"):
            if (out / f"{name}.csv").exists():
                rec[name] = read_csv_rows(out / f"{name}.csv")
        if (out / "BRIEF.md").exists():
            rec["brief"] = (out / "BRIEF.md").read_text(errors="replace")[:20000]
        if (out / "results_final.json").exists():
            rec["results_final_json"] = (out / "results_final.json").read_text(errors="replace")[:20000]
        if (out / "ANALYSIS_PATHS.jsonl").exists():
            rec["analysis_paths"] = (out / "ANALYSIS_PATHS.jsonl").read_text(errors="replace")[:300000]
        if (out / "analysis_final.py").exists():
            rec["analysis_final_py"] = (out / "analysis_final.py").read_text(errors="replace")[:20000]
        runner.usage_guard(c, rec)
    except Exception as e:  # recorded, never hidden
        rec["runner_error"] = f"{type(e).__name__}: {e}"
    rec["finished"] = time.time()
    rec["limited"] = bool(rec.get("analysis", {}).get("limit_error"))
    rec["archive"] = archive(c, cid, attempt)
    with LOCK, open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    shutil.rmtree(c, ignore_errors=True)
    return rec


def run_case(case, work_root, delay):
    time.sleep(delay)
    for attempt in range(1, MAX_TRIES + 1):
        if runner.STOP.exists():
            return {"case_id": case["case_id"], "status": "stopped"}
        rec = run_attempt(case, attempt, work_root)
        if not rec["limited"]:  # a usage-limit error is not an agent outcome, so only it is retried
            return rec
        time.sleep(random.uniform(120, 300) * attempt)
    return rec


def cmd_make_cases(a):
    with open(a.out, "w") as f:
        for run in range(1, a.runs + 1):
            for persona in PERSONAS:
                f.write(json.dumps({"case_id": secrets.token_hex(12), "persona": persona, "run": run,
                                    "delivery": a.delivery, "turns": a.turns}) + "\n")


def cmd_run(a):
    global RESULTS
    RESULTS = EXP / a.results
    runner.TIMEOUT["analysis"] = TIMEOUT_SECONDS
    if not runner.SUB_AUTH.is_file():
        raise SystemExit("subscription login missing")
    (EXP / "archives").mkdir(exist_ok=True)
    done = set()
    if RESULTS.exists():
        done = {r["case_id"] for r in map(json.loads, open(RESULTS)) if not r.get("limited")}
    cases = [c for c in map(json.loads, open(a.list)) if c["case_id"] not in done]
    work_root = Path(os.environ.get("TMPDIR", "/tmp")) / "miao-cases"
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futs = [pool.submit(run_case, c, work_root, i * a.stagger) for i, c in enumerate(cases)]
        for f in cf.as_completed(futs):
            r = f.result()
            print(json.dumps({k: r.get(k) for k in ("case_id", "persona", "run", "attempt", "limited", "status")}),
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-cases")
    m.add_argument("--runs", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--delivery", choices=["developer", "system"], default="developer")
    m.add_argument("--turns", choices=["single", "per_round"], default="single")
    r = sub.add_parser("run")
    r.add_argument("--list", required=True)
    r.add_argument("--concurrency", type=int, default=12)
    r.add_argument("--stagger", type=float, default=20.0)
    r.add_argument("--results", default="results.jsonl")
    a = ap.parse_args()
    {"make-cases": cmd_make_cases, "run": cmd_run}[a.cmd](a)
