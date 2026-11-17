#!/usr/bin/env python3
"""Originals on Claude Opus 5.5 and Sonnet 5.5: the Claude Agent SDK's CLI on the authors' Max subscription.

Same guest as the main run. The SDK's query() is a wrapper around this binary, so the runner calls
the binary directly, as card-opus-audit-20261030 did, and both routes share one Python driver. The
token is read from private/claude-token (mode 600, removed after the run), passed through the
process environment, unset in the agent's shell by clean_bash.sh, and never saved. A rejected or
warning rate-limit event stops new claims on this route.

CLAUDE_CODE_SUBPROCESS_ENV_SCRUB must stay unset: on CLI 2.1.314 it forces default permissions and
requires Claude's own sandbox, which cannot start here, so every Bash call fails and decoy .env
files appear in the workspace (found by check_real_cli.sbatch, 30 October).

  python3 run_claude.py --model claude-opus-5-5 --campaign opus-main --task 0 --tasks 2 --workers 5
"""
import argparse
import json
import os
from pathlib import Path

import common

BINARY = Path(os.environ.get("MMO_CLAUDE_BIN", common.RUN / "runtime" / "claude"))
TOKEN_FILE = common.RUN / "private" / "claude-token"
KEY_FILE = common.RUN / "private" / "anthropic-key"  # when present, the Anthropic API is used instead of the Max login (wedge runs, 1 November)
GUEST_DIR, GUEST_CLI, GUEST_SHELL, GUEST_HOME = "/opt/agent", "/opt/agent/claude", "/opt/agent/bash", "/claude-home"
TOOLS = "Bash,Read,Write,Edit,Glob,Grep"
RATE_STOP = ("rejected", "allowed_warning")


class ClaudeRoute:
    kind = "claude"
    label = "claude_max_subscription"
    start_gap = 10.0

    def __init__(self, model):
        self.model = model
        self.paused = False
        if KEY_FILE.exists():
            self.token = KEY_FILE.read_text().strip()
            self.secret_env = {"ANTHROPIC_API_KEY": self.token}
            self.label = "anthropic_api"
        else:
            self.token = TOKEN_FILE.read_text().strip()
            self.secret_env = {"CLAUDE_CODE_OAUTH_TOKEN": self.token}

    def secrets(self):
        return [self.token]

    def guest_env(self):
        return dict(common.guest_env(GUEST_DIR), HOME=common.GUEST["home"], TMPDIR=common.GUEST["tmp"],
                    CLAUDE_CONFIG_DIR=GUEST_HOME, SHELL="/bin/bash", LANG="en_US.UTF-8", DISABLE_AUTOUPDATER="1",
                    CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", CLAUDE_CODE_SHELL=GUEST_SHELL,
                    **json.loads(os.environ.get("MMO_GUEST_ENV", "{}")))  # checks/ only: points at the offline fake API

    def ro_binds(self):
        return {GUEST_CLI: BINARY, GUEST_SHELL: common.HERE / "clean_bash.sh"}

    def dirs(self):
        return (GUEST_DIR,)

    def binary_sha(self):
        return common.sha(BINARY)

    def version_commands(self):
        return {"cli_version": [GUEST_CLI, "--version"], "python": ["python3", "--version"], "r": ["R", "--version"]}

    def prepare(self, c):
        (c / "claude-home").mkdir()

    def cleanup(self, c):
        pass

    def keep(self, c):
        return [c / "claude-home" / "projects"]

    def command(self, session_id, schema):
        cmd = [GUEST_CLI, "-p", "--output-format", "stream-json", "--verbose", "--model", self.model,
               "--effort", common.EFFORT, "--tools", TOOLS, "--append-system-prompt", common.DATA_SCOPE,
               "--dangerously-skip-permissions", "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands"]
        if session_id:
            cmd += ["--resume", session_id]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema)]
        return cmd

    def parse(self, text):
        out = {"models": set(), "tools": None, "rate_events": [], "permission_denied": 0}
        result = None
        for line in text.splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            kind = e.get("type")
            if kind == "system" and e.get("subtype") == "init":
                out.update(session_id=e.get("session_id"), tools=e.get("tools"), init_model=e.get("model"),
                           cli_version=e.get("claude_code_version"), permission_mode=e.get("permissionMode"))
                out["models"].add(e.get("model"))
            elif kind == "assistant":
                out["models"].add((e.get("message") or {}).get("model"))
            elif kind == "system" and e.get("subtype") == "permission_denied":
                out["permission_denied"] += 1
            elif kind == "rate_limit_event":
                out["rate_events"].append(e.get("rate_limit_info"))
            elif kind == "result":
                result = e
        out["models"] = sorted(m for m in out["models"] if m and m != "<synthetic>")
        return out, result

    def phase(self, c, phase, prompt, session_id=None, schema=None):
        guest = common.bwrap(c, self.command(session_id, schema), env=self.guest_env(), ro_binds=self.ro_binds(),
                             dirs=self.dirs(), binds={GUEST_HOME: c / "claude-home"})
        rec = common.run_guest(c, phase, guest, prompt, secret_env=self.secret_env)
        out, result = self.parse((c / "logs" / f"{phase}.events.jsonl").read_text())
        rec.update({k: out.get(k) for k in ("session_id", "tools", "init_model", "cli_version", "models", "permission_mode",
                                            "permission_denied")})
        rec["rate_events"] = out["rate_events"][-3:]
        if any((x or {}).get("status") in RATE_STOP for x in out["rate_events"]):
            self.paused = True
        if result is None:
            rec.update(ok=False, fault_kind="timeout" if rec["timed_out"] else "cut_off", error="no result event")
            return rec
        rec.update({k: result.get(k) for k in ("subtype", "is_error", "api_error_status", "num_turns", "stop_reason",
                                               "total_cost_usd", "usage", "modelUsage", "terminal_reason")})
        if rec["permission_denied"]:  # the harness refused the agent a tool: not an agent outcome
            rec.update(ok=False, fault_kind="permission_denied", error="tool call refused by the harness")
            return rec
        if result.get("subtype") == "success" and not result.get("is_error") and not rec["timed_out"] and rec["returncode"] == 0:
            structured = result.get("structured_output")
            rec.update(ok=True, answer=json.dumps(structured) if structured is not None else result.get("result") or "")
            return rec
        text = str(result.get("result") or "") + str(result.get("errors") or "")
        status = result.get("api_error_status")
        if rec["timed_out"]:
            kind = "timeout"
        elif "limit" in text.lower() and status != 429:
            kind = "usage_limit"
            self.paused = True
        elif status == 429:
            kind = "rate_limit"
        elif status is not None:
            kind = "provider_error"
        else:
            kind = "cut_off"
        rec.update(ok=False, fault_kind=kind, error=text[-500:])
        return rec

    def facts(self, c):
        """The main-loop model as the stream recorded it; a helper's Haiku call shows only in modelUsage."""
        models = set()
        for p in (c / "logs").glob("*.events.jsonl"):
            models.update(self.parse(p.read_text())[0]["models"])
        return {"session_models": sorted(models)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("claude-opus-5-5", "claude-sonnet-5-5"))
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, help="pilot: only the first N roster positions of this model")
    a = ap.parse_args()
    common.run_cases(ClaudeRoute(a.model), a.task, a.tasks, a.workers, a.campaign, a.limit)
