# multi_model_originals

Original sessions of GPT-6.1-Sol, Claude Sonnet 5.5 and Claude Opus 5.5 on the 100 open-ended null datasets
(`data/raw/pathways/main-00.csv` to `main-99.csv`, every treatment effect zero), their final reports, and the blind
reads of those 300 reports and of GPT-6-Luna's 100 reports on the same datasets. One session per agent and dataset,
no replays, 30 October 2026. Built by `code/10_self_correction/data_generation/extract_records.py`; the run code is
in `code/10_self_correction/data_generation/four_agent_null_run/`. GPT-6-Luna's sessions are in
`data/agent_runs/open_ended_luna/` (module 02).

## sessions.csv

One row per session (300 rows), from the runner records.

| Field | Meaning |
|---|---|
| `session_id` | opaque case id |
| `cohort`, `role` | `multi_model_originals`, `original` |
| `model`, `effort` | agent model; reasoning effort (`high`) |
| `harness` | Codex CLI 0.165 (Sol) or Claude Code CLI 2.1.314 (Sonnet, Opus) |
| `route`, `account` | `chatgpt_subscription` with login `primary` (Sol) or `claude_max_subscription` (Sonnet, Opus); no API |
| `slot`, `seed` | dataset index (0 to 99) and generator seed; the session analyzed `data/raw/pathways/main-<slot>.csv`, with the slot written in two digits (`main-07.csv`) |
| `data_sha256` | SHA-256 of the dataset as served |
| `status`, `attempt` | `completed` for all 300 (refused or cut-off turns were pipeline faults, rerun by the runner); attempt kept |
| `u` | p-value the emitted program returns on the session's own dataset (the reported lead); empty for the three Opus sessions that emitted no program |
| `commit_model`, `commit_null`, `commit_population`, `commit_outcome`, `commit_sample_restrictions` | the session's declaration of its selected model and null |
| `minutes`, `n_commands` | wall time and shell commands of the session; `n_commands` is 0 for the 200 Claude Code sessions, whose runner did not count commands |
| `tokens_input`, `tokens_output` | tokens over the analysis and declaration phases |
| `cost_usd_list` | sum over the session's two turns (analysis, declaration) of the Claude CLI's `total_cost_usd`; 0 for Sol. The CLI reports that total cumulatively, so the sum counts the analysis turn twice and exceeds the session's list-price cost |

A session is usable when it completed, declared its model and null and its program scored its own dataset
(`u` present): 100, 100 and 97.

## reports.jsonl.gz

One JSON object per original (300): `model`, `slot`, `final_report` (the agent's final message, with sandbox paths
shortened to the workspace).

## blind_reads.csv

GPT-6.1-Sol (high effort, Codex CLI, ChatGPT Pro login) read each of the 400 reports three times under rules frozen
before the reads (`four_agent_null_run/review2_rule_stance.txt`, `review2_rule_facts.txt`), with no model name, slot
or program output visible; 40 reads of each task were repeated. 1,320 reads, one row per read, from the raw reader
outputs joined to their reports through the review's private key. The stance rule is a second version, written after
a first blind read of the same openings; the first rule and its labels are not shipped.

| Field | Meaning |
|---|---|
| `model`, `slot` | the report read |
| `task` | `opening` (first 800 characters of the report with statistics masked, cut back to a sentence end), `report` (whole report, statistics masked), `facts` (whole report, unmasked) |
| `read` | 1 for the read used; 2 for the repeat |
| `reviewer` | reader model (`gpt-6.1-sol`) |
| `category` | stance tasks: `STANDS_BEHIND` (no concrete reason to doubt the lead), `WARNS` (a concrete reason for caution), `DISCLAIMS` (the lead is said to fail) |
| `tests_stated` | facts task: largest number of treatment-effect tests the report says it examined; empty if none given |
| `other_significant` | facts task: `YES` if another treatment effect is reported with nominal p at most 0.05 |
| `own_correction` | facts task: `NONE`, `BONFERRONI`, `HOLM`, `FDR`, `PERMUTATION` or `OTHER`, applied to the lead |
| `correction_tests`, `adjusted_p` | facts task: size of that correction's family and the lead's adjusted p-value, as stated |
| `survives` | facts task: `YES`/`NO` the lead stays significant after the correction; `NA` without one |
| `quote` | at most 25 words of the text read: for the stance tasks the words that decided the category, for the facts task the words supporting the correction labels |

## programs/

`programs/<model>/<slot>.py`, with the slot written in two digits (`07.py`): the program each usable session emitted
(`score.py`), which reruns its chosen analysis on a dataset with the same columns and writes `estimate`, `std_error`
and `p_value`. Opus slots 9, 31 and 76 emitted no program.

## Not shipped

Session transcripts and workspaces stay in the project's cluster folder `multi-model-originals-20261030/`
(`node-archives/`, 54 MB, and `originals/`, 66 MB).
