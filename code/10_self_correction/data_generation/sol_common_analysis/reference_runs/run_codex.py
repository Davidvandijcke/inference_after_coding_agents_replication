#!/usr/bin/env python3
"""Originals on gpt-6.1-terra and gpt-6.1-sol: Codex CLI on a ChatGPT Pro login, one session per dataset.

The guest is the main run's. Each case gets a fresh Codex home and its own copy of the login with
any API key removed; the copy is deleted when the agent exits. An account stops claiming cases when
its weekly meter is full (100%; 85% before 31 October, 90% before 1 November) or when a turn is refused
at the limit; a refused turn is a pause, counted as not done and rerun after the reset. ChatGPT credits
are never drawn.

  python3 run_codex.py --model gpt-6.1-sol --account primary --campaign sol-main --task 0 --tasks 2 --workers 10
"""
import argparse
import json
import os
from pathlib import Path
import re

import common

G = Path("<cluster-project>")
CLI_DIR = Path(os.environ.get("MMO_CODEX_DIR", G / "codex-cli-0.165.0/node_modules/@openai/codex-linux-x64/vendor"
                              "/x86_64-unknown-linux-musl/bin"))
AUTH_ROOT = Path(os.environ.get("MMO_CODEX_AUTH", G / "sol-card-main-gl/auth"))
GUEST_DIR, GUEST_CLI, GUEST_CMH, GUEST_HOME = "/opt/agent", "/opt/agent/codex", "/opt/agent/codex-code-mode-host", "/codex-home"
# This bounded run starts from the live account meter, without rebasing.  Its
# absolute ceiling is 77%; 97% remains an independent account-wide emergency
# ceiling. Both named meter windows are checked.
PAUSE_AT = float(os.environ.get("MMO_RUN_PAUSE_AT", "77"))
ACCOUNT_WIDE_PAUSE_AT = float(os.environ.get("MMO_ACCOUNT_PAUSE_AT", "97"))
USAGE_TEXT = ("usage_limit", "usage limit", "hit your usage limit", "rate_limit_reached")
RATE_PATTERN = re.compile(r"rate.limit|too many requests|\b429\b|quota", re.I)
DISABLED_FEATURES = ("apps", "auth_elicitation", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
                     "computer_use", "fast_mode", "goals", "guardian_approval", "hooks", "image_generation",
                     "in_app_browser", "in_app_chat", "in_app_dictation", "in_app_updates", "memories", "multi_agent",
                     "multi_agent_v2", "plugin_sharing", "plugins", "remote_plugin", "shell_snapshot",
                     "skill_mcp_dependency_install", "skill_search", "tool_call_mcp_elicitation", "tool_suggest",
                     "unbounded_connection_retries", "view_image", "workspace_dependencies")


class CodexRoute:
    kind = "codex"
    start_gap = 5.0

    def __init__(self, model, account):
        self.model, self.account = model, account
        self.label = f"chatgpt_subscription:{account}"
        self.paused = False
        self.login = json.loads((AUTH_ROOT / account / "auth.json").read_text())
        if self.login.get("auth_mode") != "chatgpt":
            raise SystemExit("ChatGPT subscription login required")
        self.login["OPENAI_API_KEY"] = None
        tokens = self.login.get("tokens") or {}
        self._secrets = [v for v in tokens.values() if isinstance(v, str) and len(v) > 20]

    def secrets(self):
        return self._secrets

    def guest_env(self):
        return dict(common.guest_env(GUEST_DIR), HOME=common.GUEST["home"], CODEX_HOME=GUEST_HOME,
                    TMPDIR=common.GUEST["tmp"])

    def ro_binds(self):
        return {GUEST_CLI: CLI_DIR / "codex", GUEST_CMH: CLI_DIR / "codex-code-mode-host"}

    def dirs(self):
        return (GUEST_DIR,)

    def binary_sha(self):
        return common.sha(CLI_DIR / "codex")

    def version_commands(self):
        return {"cli_version": [GUEST_CLI, "--version"], "python": ["python3", "--version"], "r": ["R", "--version"]}

    def config_text(self):
        """The main run's configuration on CLI 0.165, with Codex's default provider as in the later subscription runs."""
        v = {"model": self.model, "model_reasoning_effort": common.EFFORT, "approval_policy": "never",
             "sandbox_mode": "danger-full-access", "web_search": "disabled", "project_doc_max_bytes": 0,
             "project_doc_fallback_filenames": [], "shell_environment_policy.inherit": "none",
             "shell_environment_policy.experimental_use_profile": False,
             "shell_environment_policy.ignore_default_excludes": False, "memories.generate_memories": False,
             "memories.use_memories": False, "developer_instructions": common.DATA_SCOPE,
             "forced_login_method": "chatgpt", "cli_auth_credentials_store": "file"}
        for k, val in common.guest_env(GUEST_DIR).items():
            v[f"shell_environment_policy.set.{k}"] = val
        for k, val in (("HOME", common.GUEST["home"]), ("TMPDIR", common.GUEST["tmp"])):
            v[f"shell_environment_policy.set.{k}"] = val
        v.update({f"features.{n}": False for n in DISABLED_FEATURES})
        v.update({"features.code_mode_host": True, "features.shell_tool": True, "features.unified_exec": True})
        text = "\n".join(f"{k} = {json.dumps(x)}" for k, x in sorted(v.items())) + "\n"
        return os.environ.get("MMO_CODEX_EXTRA_CONFIG", "") + text  # checks/ only: points at the offline fake API

    def prepare(self, c):
        home = c / "codex-home"
        home.mkdir()
        (home / "config.toml").write_text(self.config_text())
        auth = home / "auth.json"
        auth.write_text(json.dumps(self.login))
        auth.chmod(0o600)

    def cleanup(self, c):
        (c / "codex-home" / "auth.json").unlink(missing_ok=True)

    def keep(self, c):
        return [c / "codex-home" / "sessions"]

    def parse(self, text, expected=None):
        events = []
        for line in text.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
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
        return {"session_id": session, "answer": answers[-1], "usage": usage[-1] if usage else None, "n_events": len(events),
                "n_commands": sum(1 for e in events if e.get("type") == "item.completed"
                                  and e.get("item", {}).get("type") in ("command_execution", "function_call",
                                                                         "custom_tool_call", "local_shell_call"))}

    def phase(self, c, phase, prompt, session_id=None, schema=None):
        cmd = [GUEST_CLI, "exec", "--cd", common.GUEST["workspace"], "--strict-config", "--ignore-rules",
               "--model", self.model, "--skip-git-repo-check"]
        if session_id:
            cmd += ["resume", session_id]
        cmd += ["--json"]
        schema_file = None
        if schema is not None:
            schema_file = c / f"{phase}-schema.json"
            schema_file.write_text(json.dumps(schema))
            cmd += ["--output-schema", common.GUEST_SCHEMA]
        cmd += ["-"]
        guest = common.bwrap(c, cmd, env=self.guest_env(), ro_binds=self.ro_binds(), dirs=self.dirs(), schema=schema_file,
                             binds={GUEST_HOME: c / "codex-home"})
        rec = common.run_guest(c, phase, guest, prompt)
        events = (c / "logs" / f"{phase}.events.jsonl").read_text()
        try:
            rec.update(self.parse(events, session_id))
            rec["ok"] = not rec["timed_out"] and rec["returncode"] == 0
        except ValueError as e:
            rec.update(ok=False, error=str(e))
        if not rec["ok"]:
            text = events + (c / "logs" / f"{phase}.stderr").read_text()[-4000:]
            if rec["timed_out"]:
                rec["fault_kind"] = "timeout"
            elif any(s in text for s in USAGE_TEXT):
                rec["fault_kind"] = "usage_limit"
                self.paused = True
            elif RATE_PATTERN.search(text):
                rec["fault_kind"] = "rate_limit"
            else:
                rec["fault_kind"] = "cut_off"
        return rec

    def facts(self, c):
        """Model and effort as the session recorded them, and the account's meter."""
        models, efforts, latest = set(), set(), None
        for f in (c / "codex-home" / "sessions").rglob("*.jsonl"):
            for line in f.read_text(errors="replace").splitlines():
                i = line.find('"rate_limits":')
                if i >= 0:
                    try:
                        snap, _ = json.JSONDecoder().raw_decode(line[i + len('"rate_limits":'):].lstrip())
                        latest = snap or latest
                    except ValueError:
                        pass
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                if x.get("type") == "turn_context":
                    models.add(x["payload"].get("model"))
                    efforts.add(x["payload"].get("effort"))
        out = {"session_models": sorted(m for m in models if m), "session_efforts": sorted(e for e in efforts if e)}
        if latest:
            # Subscription sessions expose more than one meter window.  Guard
            # every named window, since `primary` is not guaranteed to be the
            # weekly meter.  Keep only non-secret meter facts in the receipt.
            windows = {}
            for name in ("primary", "secondary"):
                window = latest.get(name) or {}
                windows[name] = {k: window.get(k) for k in
                                 ("used_percent", "window_minutes", "resets_at") if window.get(k) is not None}
            out["meter_windows"] = windows
            used_values = [float(w["used_percent"]) for w in windows.values() if w.get("used_percent") is not None]
            if (any(used >= PAUSE_AT for used in used_values)
                    or any(used >= ACCOUNT_WIDE_PAUSE_AT for used in used_values)
                    or latest.get("rate_limit_reached_type")
                    or latest.get("spend_control_reached")):
                self.paused = True
                # Each account's two model lanes share this symlink target, so
                # either job stops new claims for that subscription account.
                (common.RUN / "STOP-codex").write_text(json.dumps({
                    "reason": "subscription meter guard", "run_threshold": PAUSE_AT,
                    "account_wide_threshold": ACCOUNT_WIDE_PAUSE_AT,
                    "meter_windows": windows, "time": __import__("time").time(),
                }))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-luna"))
    ap.add_argument("--account", required=True, choices=("primary", "second"))
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, help="pilot: only the first N roster positions of this model")
    a = ap.parse_args()
    common.run_cases(CodexRoute(a.model, a.account), a.task, a.tasks, a.workers, a.campaign, a.limit)
