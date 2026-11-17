#!/usr/bin/env node
// Replication of Miao et al.'s ISSP persona experiment with their harness settings
// (codes/agentic_bootstrap.mjs at commit 7c8925f, command B in examples/issp/README.md).
//
// Same as theirs: Claude Agent SDK query(), persona text as the whole systemPrompt, the
// claude_code tool preset, maxTurns 200, bypassPermissions, their sandbox block, model
// claude-sonnet-4-6, their task prompt (data prompt + 10 rounds x 10 models over 7 axes
// + final selection), output dir <root>/examples/issp/results/<persona>/run_<id>.
// Deliberate differences (README.md): the paper's sign convention in the data prompt;
// each run gets its own fresh root holding only the data, so runs cannot see each other's
// results or the repository's README; a minimal environment carrying the subscription token.
//
// Usage: node run_miao.mjs --personas anti_immigration,pro_immigration,neutral --runs 1 \
//          [--start 1] [--concurrency 3] [--model claude-sonnet-4-6] [--rounds 10] [--results results]
//          [--data data_clean.csv (file in <runtime>/data, copied into the run as data_clean.csv)] [--data_prompt issp_paper.md]
//          [--route subscription|openrouter]

import { cpSync, constants, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync, readdirSync, statSync, unlinkSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { randomBytes } from "crypto";
import { gzipSync } from "zlib";

const SRC = dirname(fileURLToPath(import.meta.url));
const EXP = dirname(SRC);
const PROMPTS = join(SRC, "prompts");
const RT = process.env.MIAO_RUNTIME || "<local runtime>";
const { query } = await import(join(RT, "node_modules/@anthropic-ai/claude-agent-sdk/sdk.mjs"));
const QUESTION = "How does immigration affect public support for social welfare programs?";
const ENV_FILE = join(EXP, "..", "..", ".env");

function args(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) if (argv[i].startsWith("--")) a[argv[i].slice(2)] = argv[i + 1];
  return a;
}

// Subscription route: the authors' Claude subscription token. OpenRouter route: the OpenRouter key against
// OpenRouter's Anthropic-compatible endpoint (billed per token; prompt caching passes through).
function token(name) {
  const line = readFileSync(ENV_FILE, "utf-8").split("\n").find((l) => l.startsWith(`${name}=`));
  return line.slice(name.length + 1).trim().replace(/^["']|["']$/g, "");
}

function authEnv(cfg) {
  return cfg.route === "openrouter"
    ? { ANTHROPIC_BASE_URL: "https://openrouter.ai/api", ANTHROPIC_AUTH_TOKEN: cfg.token, ANTHROPIC_API_KEY: "" }
    : { CLAUDE_CODE_OAUTH_TOKEN: cfg.token };
}

const read = (name) => readFileSync(join(PROMPTS, name), "utf-8").trimEnd();

// Same assembly as their prompts/load_prompts.mjs.
function taskPrompt(dataPath, outDir, numRounds, dataPrompt = "issp_paper.md") {
  const data = read(dataPrompt).replaceAll("{{DATA_PATH}}", dataPath).replaceAll("{{RESEARCH_QUESTION}}", QUESTION);
  const [intro, rest] = read("_round_template.md").split("%%REPEAT_FOR_R_FROM_2_TO_NUM_ROUNDS%%");
  const [loopBody, outro] = rest.split("%%END_REPEAT%%");
  let rounds = intro.replaceAll("{{OUT_DIR}}", outDir);
  for (let r = 2; r <= numRounds; r++)
    rounds += loopBody.replaceAll("{{R}}", String(r)).replaceAll("{{R_MINUS_1}}", String(r - 1)).replaceAll("{{OUT_DIR}}", outDir);
  rounds += outro || "";
  const n = Array.from({ length: numRounds }, (_, i) => i + 1);
  const instructions = read("10round_7axis.md")
    .replaceAll("{{NUM_ROUNDS}}", String(numRounds))
    .replaceAll("{{OUT_DIR}}", outDir)
    .replaceAll("{{SCRIPT_FILES}}", n.map((i) => `analysis_r${i}.py`).join(", "))
    .replaceAll("{{RESULT_FILES}}", n.map((i) => `results_r${i}.csv`).join(", "))
    .replaceAll("{{ROUND_INSTRUCTIONS}}", rounds);
  return `${data}\n\n${instructions}\n`;
}

// True if another driver (including ones started before claims existed) has this run open: a workspace with
// the same persona/run path whose message stream is on the same model and was written in the last 15 minutes.
function busyElsewhere(persona, runId, model, results) {
  for (const key of readdirSync(join(RT, "ws"))) {
    const log = join(RT, "ws", key, "messages.jsonl");
    if (!existsSync(join(RT, "ws", key, "AgenticBootstrap", "examples", "issp", "results", persona, `run_${runId}`)) || !existsSync(log)) continue;
    const arm = join(RT, "ws", key, "arm.txt");  // workspaces from before arm markers belong to the original arm
    if ((existsSync(arm) ? readFileSync(arm, "utf-8").trim() : "results") !== results) continue;
    if (Date.now() - statSync(log).mtimeMs > 15 * 60 * 1000) continue;
    const first = readFileSync(log, "utf-8").split("\n", 1)[0];
    if (first && JSON.parse(first).model === model) return true;
  }
  return false;
}

async function runOne({ persona, run, cfg }) {
  const runId = String(run).padStart(2, "0");
  const keep = join(EXP, cfg.results, persona, `run_${runId}`);
  // A run is claimed by a placeholder metadata.json, so several drivers can share one range.
  if (existsSync(join(keep, "metadata.json")) || busyElsewhere(persona, runId, cfg.model, cfg.results)) return { persona, run, skipped: true };
  mkdirSync(keep, { recursive: true });
  writeFileSync(join(keep, "metadata.json"), JSON.stringify({ status: "running", model: cfg.model, started: new Date().toISOString() }));
  const root = join(RT, "ws", randomBytes(4).toString("hex"), "AgenticBootstrap");
  const dataDir = join(root, "examples", "issp", "data");
  const dataPath = join(dataDir, "data_clean.csv");
  const outDir = join(root, "examples", "issp", "results", persona, `run_${runId}`);
  mkdirSync(dataDir, { recursive: true });
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(dirname(root), "arm.txt"), cfg.results + "\n");
  copyFileSync(join(RT, "data", cfg.data), dataPath, constants.COPYFILE_FICLONE);
  const log = join(dirname(root), "messages.jsonl");

  const env = {
    HOME: process.env.HOME, USER: process.env.USER, LANG: "en_US.UTF-8", TMPDIR: "/tmp",
    PATH: `${join(RT, "venv", "bin")}:${dirname(process.execPath)}:/usr/bin:/bin:/usr/sbin:/sbin`,
    CLAUDE_CONFIG_DIR: join(RT, "claude-config"),  // keeps these sessions out of the authors' own ~/.claude
    ...authEnv(cfg),
  };
  console.log(`[${persona}] run ${runId} - starting (${root})`);
  const t0 = Date.now();
  let fullOutput = "", costUsd = 0, resultMeta = {}, init = {}, prompt = taskPrompt(dataPath, outDir, cfg.rounds, cfg.dataPrompt);
  let sessionId, resumeId, resumes = 0;
  // A transient API error (e.g. OpenRouter's 402 when in-flight requests exceed the prepaid balance) ends the
  // session without a result; resume the same session instead of discarding the run, and record each resume.
  for (;;) {
  let lastResult = "";
  try {
    const conv = query({
      prompt,
      options: {
        ...(resumeId ? { resume: resumeId } : {}),
        systemPrompt: read(`${persona}.md`).trim(),
        cwd: root,
        tools: { type: "preset", preset: "claude_code" },
        maxTurns: 200,
        permissionMode: "bypassPermissions",
        allowDangerouslySkipPermissions: true,
        model: cfg.model,
        env,
        sandbox: {
          enabled: true,
          failIfUnavailable: false,
          autoAllowBashIfSandboxed: true,
          allowUnsandboxedCommands: false,
          filesystem: {
            allowWrite: [outDir, "/tmp"],
            allowRead: [outDir, dataDir],
            denyWrite: [dataDir],
            denyRead: [process.env.HOME, "~/.ssh", "~/.aws", "~/.claude"],
          },
          // deniedDomains is ours: on SDK 0.3.197 their rule alone leaves the network open (README.md).
          network: { allowedDomains: [], deniedDomains: ["*"], allowManagedDomainsOnly: true },
        },
      },
    });
    for await (const m of conv) {
      appendFileSync(log, JSON.stringify(m) + "\n");
      if (m.type === "rate_limit_event" && m.rate_limit_info?.status === "rejected") cfg.stop = true;
      if (m.type === "system" && m.subtype === "init") { init = { model: m.model, claude_code_version: m.claude_code_version, tools: m.tools }; sessionId = m.session_id; }
      if (m.type === "assistant") for (const b of m.message.content) if (b.type === "text") fullOutput += b.text + "\n";
      if (m.type === "result") {
        resultMeta = { subtype: m.subtype, is_error: m.is_error, num_turns: m.num_turns, usage: m.usage };
        lastResult = String(m.result ?? "");
        if (m.subtype === "success") { fullOutput += "\n" + m.result + "\n"; costUsd += m.total_cost_usd || 0; }
      }
    }
  } catch (err) {
    fullOutput += `\nERROR: ${err.message}\n`;
    resultMeta.error = err.message;
    lastResult = err.message;
  }
  if (!/API Error: (402|429|500|502|503|529)/.test(lastResult) || !sessionId || resumes >= 6
      || existsSync(join(outDir, "results_final.csv"))) break;
  resumes++;
  resumeId = sessionId;
  prompt = "Continue.";
  console.log(`[${persona}] run ${runId} - transient API error, resume ${resumes} after a wait`);
  await new Promise((r) => setTimeout(r, 60000 * resumes));
  }
  resultMeta.resumes = resumes;

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const roundStatus = {};
  for (let r = 1; r <= cfg.rounds; r++) roundStatus[`has_r${r}`] = existsSync(join(outDir, `results_r${r}.csv`));
  const hasFinal = existsSync(join(outDir, "results_final.csv"));
  writeFileSync(join(outDir, "full_output.txt"), fullOutput);
  // Keep the agent's deliverables plus the full message stream; scrub the token if it ever appears.
  mkdirSync(keep, { recursive: true });
  cpSync(outDir, keep, { recursive: true });
  let scrubbed = 0;
  for (const f of readdirSync(keep, { recursive: true })) {
    const p = join(keep, String(f));
    if (!/\.(csv|py|txt|md|json|log)$/.test(p)) continue;
    const s = readFileSync(p, "utf-8");
    if (s.includes(cfg.token)) { writeFileSync(p, s.replaceAll(cfg.token, "<redacted>")); scrubbed++; }
  }
  const stream = readFileSync(log, "utf-8");
  if (stream.includes(cfg.token)) scrubbed++;
  writeFileSync(join(keep, "messages.jsonl.gz"), gzipSync(stream.replaceAll(cfg.token, "<redacted>")));
  // Only a run with a final result counts as done; anything else is kept under another name and retried on rerun.
  if (!hasFinal) unlinkSync(join(keep, "metadata.json"));
  writeFileSync(join(keep, hasFinal ? "metadata.json" : `metadata_incomplete_${Date.now()}.json`), JSON.stringify({
    persona, run, model: cfg.model, route: cfg.route, num_rounds: cfg.rounds, elapsed_seconds: +elapsed, cost_usd_equivalent: costUsd,
    timestamp: new Date().toISOString(), workspace: root, ...init, ...resultMeta, ...roundStatus, has_final: hasFinal,
    token_scrubbed_files: scrubbed,
  }, null, 2));
  console.log(`[${persona}] run ${runId} - done ${elapsed}s final=${hasFinal} turns=${resultMeta.num_turns} ${resultMeta.subtype || resultMeta.error || ""}`);
  return { persona, run, elapsed, costUsd, hasFinal };
}

async function main() {
  const a = args(process.argv.slice(2));
  const route = a.route || "subscription";
  const cfg = { model: a.model || "claude-sonnet-4-6", rounds: parseInt(a.rounds || "10", 10), route,
    token: token(route === "openrouter" ? "OPENROUTER_API_KEY" : "CLAUDE_SETUP_TOKEN"), results: a.results || "results",
    data: a.data || "data_clean.csv", dataPrompt: a.data_prompt || "issp_paper.md" };
  const personas = (a.personas || "anti_immigration,pro_immigration,neutral").split(",");
  if (a.dry_run) {
    console.log(`===== SYSTEM PROMPT (${personas[0]}) =====\n${read(`${personas[0]}.md`).trim()}\n\n===== TASK PROMPT =====\n` +
      taskPrompt("<root>/examples/issp/data/data_clean.csv", `<root>/examples/issp/results/${personas[0]}/run_01`, cfg.rounds, cfg.dataPrompt));
    return;
  }
  const start = parseInt(a.start || "1", 10), runs = parseInt(a.runs || "1", 10);
  const tasks = [];
  for (let i = start; i < start + runs; i++) for (const persona of personas) tasks.push({ persona, run: i, cfg });
  if (a.reverse) tasks.reverse();
  let idx = 0;
  const results = [];
  async function worker() { while (idx < tasks.length && !cfg.stop) results.push(await runOne(tasks[idx++])); }
  await Promise.all(Array.from({ length: Math.min(parseInt(a.concurrency || "4", 10), tasks.length) }, worker));
  console.log(`Done: ${results.filter((r) => r.hasFinal).length}/${results.length} with a final result${cfg.stop ? "; stopped on a usage limit" : ""}`);
}

main().catch((err) => { console.error("Fatal:", err.message); process.exit(1); });
