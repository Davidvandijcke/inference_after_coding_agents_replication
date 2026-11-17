#!/usr/bin/env python3
"""Miao et al.'s persona design with GPT-6-Luna through the Codex CLI on the authors' ChatGPT subscription (29 October).

Same task prompt as run_miao.mjs (their data prompt, ten rounds of ten models over seven axes, final selection),
assembled from the same files in prompts/, the same persona files, and the same data file in a fresh
<tmp>/AgenticBootstrap/examples/issp/data/ per run. As in the earlier GPT-6-Luna exact-design runs (arm B,
../miao-luna-persona-pilot-20261026/source/miao_pilot.py), the persona replaces Codex's built-in instructions
(model_instructions_file), as Miao et al.'s systemPrompt replaces Claude Code's; gpt-6-luna at high effort. Each run
has its own CODEX_HOME holding only a copy of the login and a config with project docs, memories, web search,
subagents and plugins off; Codex's macOS sandbox allows writes only in the run folder and /tmp and blocks the
network. Python is the replication's venv (pandas, numpy, scipy, statsmodels, scikit-learn, formulaic).
Outputs go to <results>/<persona>/run_<id>/ in the layout of the Sonnet arms, with metadata.json and the event
stream. Usage: python run_luna_codex.py --personas anti_immigration,pro_immigration,neutral --runs 10
          [--start 1] [--concurrency 6] [--data issp_clean.csv] [--data_prompt issp_paper_clean.md]
          [--results results_gpt-6-luna_clean]
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = Path(__file__).resolve().parent
EXP = SRC.parent
PROMPTS = SRC / "prompts"
RT = Path(os.environ.get("MIAO_RUNTIME", "<local runtime>"))
CODEX = "<Codex CLI binary>"
MODEL, EFFORT = "gpt-6-luna", "high"
QUESTION = "How does immigration affect public support for social welfare programs?"
TIMEOUT = 2 * 3600
LOCK = threading.Lock()
OFF = ("apps", "browser_use", "computer_use", "fast_mode", "goals", "hooks", "image_generation", "memories",
       "multi_agent", "multi_agent_v2", "personality", "plugins", "remote_plugin", "skill_search", "tool_suggest",
       "view_image")


def read(name):
    return (PROMPTS / name).read_text().rstrip()


def task_prompt(data_path, out_dir, rounds=10, data_prompt="issp_paper_clean.md"):
    """Port of taskPrompt in run_miao.mjs (their prompts/load_prompts.mjs assembly)."""
    data = read(data_prompt).replace("{{DATA_PATH}}", data_path).replace("{{RESEARCH_QUESTION}}", QUESTION)
    intro, rest = read("_round_template.md").split("%%REPEAT_FOR_R_FROM_2_TO_NUM_ROUNDS%%")
    loop, outro = (rest.split("%%END_REPEAT%%") + [""])[:2]
    text = intro.replace("{{OUT_DIR}}", out_dir)
    for r in range(2, rounds + 1):
        text += loop.replace("{{R}}", str(r)).replace("{{R_MINUS_1}}", str(r - 1)).replace("{{OUT_DIR}}", out_dir)
    text += outro or ""
    n = range(1, rounds + 1)
    instructions = (read("10round_7axis.md").replace("{{NUM_ROUNDS}}", str(rounds)).replace("{{OUT_DIR}}", out_dir)
                    .replace("{{SCRIPT_FILES}}", ", ".join(f"analysis_r{i}.py" for i in n))
                    .replace("{{RESULT_FILES}}", ", ".join(f"results_r{i}.csv" for i in n))
                    .replace("{{ROUND_INSTRUCTIONS}}", text))
    return f"{data}\n\n{instructions}\n"


def config(persona_file):
    v = {"model": MODEL, "model_reasoning_effort": EFFORT, "service_tier": "default", "approval_policy": "never",
         "sandbox_mode": "workspace-write", "sandbox_workspace_write.network_access": False,
         "web_search": "disabled", "project_doc_max_bytes": 0, "agents.enabled": False,
         "model_instructions_file": str(persona_file),
         "shell_environment_policy.inherit": "core",
         "shell_environment_policy.set.PATH": f"{RT / 'venv' / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin"}
    v.update({f"features.{f}": False for f in OFF})
    return "\n".join(f"{k} = {json.dumps(x)}" for k, x in sorted(v.items())) + "\n"


def run_one(persona, run, a):
    rid = f"{run:02d}"
    keep = EXP / a.results / persona / f"run_{rid}"
    if (keep / "metadata.json").exists():
        return
    keep.mkdir(parents=True, exist_ok=True)
    (keep / "metadata.json").write_text(json.dumps({"status": "running", "model": MODEL}))
    base = Path(tempfile.mkdtemp(prefix="luna-miao-"))
    root, home = base / "AgenticBootstrap", base / "codex-home"
    data_dir, out_dir = root / "examples" / "issp" / "data", root / "examples" / "issp" / "results" / persona / f"run_{rid}"
    for d in (data_dir, out_dir, home):
        d.mkdir(parents=True)
    shutil.copy(RT / "data" / a.data, data_dir / "data_clean.csv")
    shutil.copy(Path.home() / ".codex" / "auth.json", home / "auth.json")
    (home / "persona.md").write_text(read(f"{persona}.md").strip() + "\n")
    (home / "config.toml").write_text(config(home / "persona.md"))
    prompt = task_prompt(str(data_dir / "data_clean.csv"), str(out_dir), data_prompt=a.data_prompt)
    with LOCK:
        print(f"[{persona}] run {rid} - starting ({root})", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run([CODEX, "exec", "--skip-git-repo-check", "-C", str(root), "--json", "-"], input=prompt,
                           capture_output=True, text=True, env={**os.environ, "CODEX_HOME": str(home)}, timeout=TIMEOUT)
        stdout, err = p.stdout, p.stderr[-2000:]
    except subprocess.TimeoutExpired as e:
        stdout, err = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
    events = [json.loads(x) for x in stdout.splitlines() if x.startswith("{")]
    usage = next((e["usage"] for e in reversed(events) if "usage" in e), {})
    for f in out_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, keep / f.name)
    (keep / "events.jsonl.gz").write_bytes(gzip.compress(stdout.encode()))
    meta = {"model": MODEL, "effort": EFFORT, "route": "codex-subscription", "persona": persona, "run": run,
            "data": a.data, "data_prompt": a.data_prompt, "elapsed_seconds": round(time.time() - t0),
            "usage": usage, "has_final": (keep / "results_final.csv").exists(),
            **{f"has_r{k}": (keep / f"results_r{k}.csv").exists() for k in range(1, 11)},
            "failed": [e for e in events if e.get("type") in ("turn.failed", "error")][:3], "stderr_tail": err[-500:],
            "codex_version": subprocess.run([CODEX, "--version"], capture_output=True, text=True).stdout.strip()}
    (keep / "metadata.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(base, ignore_errors=True)
    with LOCK:
        print(f"[{persona}] run {rid} - done in {meta['elapsed_seconds'] / 60:.0f} min, final={meta['has_final']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="anti_immigration,pro_immigration,neutral")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--data", default="issp_clean.csv")
    ap.add_argument("--data_prompt", default="issp_paper_clean.md")
    ap.add_argument("--results", default="results_gpt-6-luna_clean")
    a = ap.parse_args()
    jobs = [(p, r) for r in range(a.start, a.start + a.runs) for p in a.personas.split(",")]
    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(lambda j: run_one(*j, a), jobs))


if __name__ == "__main__":
    main()
