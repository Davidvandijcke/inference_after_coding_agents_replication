# card_luna: Card-Krueger application with GPT-6-Luna

One cohort, never pooled with another: one original run of GPT-6-Luna (OpenAI, high reasoning effort, Codex CLI
0.157.0, through the OpenAI API) on all 410 stores, 16 October 2026, and 3,000 replays of the same agent on reference
subsets of 150 stores (1,142 tail, 1,858 body), 17 October 2026, on the University of Michigan's Great Lakes cluster.
The files are author-generated records of agent sessions, written by
`code/08_card/data_generation/extract_records.py`, except `packet/data.csv`, which holds Card and Krueger's records in
another format and keeps the status of `data/raw/card_krueger/`. Modules 09 and 11 read these files.

Every session started from the files in `packet/` (with `data.csv` replaced by the replay's subset), an empty home
folder and a fixed Codex configuration (model `gpt-6-luna`, reasoning effort high, default service tier, no approval
prompts, full access inside the session's own container, web search off, internet through a proxy for documentation
and packages, R 4.5.1 with a tidyverse library and Python 3.12), with the operator instruction in
`prompts/developer-instructions.txt`. Each session had three turns in one conversation: analysis
(`packet/research-prompt.txt`; the agent saves its chosen analysis as a function `analysis(data)` in `analysis.R`),
commitment (`packet/commitment_prompt.txt`; it declares the model and null of that analysis as JSON with fields model,
null, population, outcome, sample_restrictions) and, for replays, comparison (`packet/comparison_prompt.txt` with the
original's declaration inserted; the replay's own verdict on whether its commitment matches).

## packet/

The files every Card agent received (the cohorts of module 09 received the same packet).

| File | Contents |
|---|---|
| `data.csv` | The 410 stores, both survey waves, 47 columns: Card and Krueger's `public.dat` with the codebook's names in lower case, missing entries as `NA`, and `store_id` (state and sheet number); built from `data/raw/card_krueger/` by `code/08_card/01_original.R`, which checks it byte for byte. Replays received 150 of these rows in a random order (`data/intermediate/08_card/`). |
| `CODEBOOK.md`, `DESIGN.md` | The variable definitions and the study setting given to the agent. |
| `research-prompt.txt` | The research request (analysis turn). |
| `commitment_prompt.txt` | The commitment request (second turn). |
| `comparison_prompt.txt` | The comparison request (third turn of a replay); `{{TARGET_JSON}}` is replaced by the original's declaration. |

## original.json, original/, programs/original-analysis.R

`original.json`: `session_id` (Codex session id), `cohort`, `role` (`original`), `model`, `effort`, `route` (`api`),
`p_value` (u = 0.0252, the p-value of its emitted program on the full data, the threshold between tail and body),
`estimate` and `se` (2.97 and 1.32 FTE per store), `commitment` (its declaration, the review target) and
`final_report` (its last message). The files the original wrote, rebuilt from the tool calls of its final model
request: `programs/original-analysis.R` (its emitted program: OLS of the change in nonmanager FTE on a New Jersey
indicator and chain indicators, HC2 standard errors, complete cases; the fixed p-value procedure run on every
reference subset), `original/analysis_report.md` (its report) and `original/explore.R` (its exploration script).

## replays.csv.gz

One row per rostered replay, in run order (gzip-compressed CSV, UTF-8). All 3,000 completed.

| Field | Meaning |
|---|---|
| session_id | Opaque case id of the replay (`card3000-` and 24 hex characters); the same id names it in every file here. |
| cohort | `card_luna`. |
| role | `replay`. |
| model, effort, route | `gpt-6-luna`, `high`, `api` (the OpenAI API, default tier). |
| position | Position in the run order: the 1,142 tail subsets, then the 1,858 body subsets, each in a seeded random order. |
| bank_id | Id of the replay's reference subset in `data/intermediate/08_card/` (1 to 20,000). |
| bank_p | p-value of the original's program on that subset. |
| stratum | `tail` when bank_p is at most u (`original.json`, `p_value`), else `body`. |
| status | `completed` for every replay. |
| commit_model, commit_null, commit_population, commit_outcome, commit_sample_restrictions | The replay's committed declaration of its selected analysis, verbatim. |
| own_match | The replay's own verdict, after seeing the original's declaration, on whether it selected the same model and null (`True`/`False`). |
| own_reason | The replay's reason for its verdict. |
| review_label | The fact label: `match`, `nonmatch` or `unresolved`, from GPT-6.1-Sol (high effort, Codex CLI 0.165.0, ChatGPT Pro subscription logins, 30 October 2026), which compared the two declarations in batches of 60 blinded pairs, without stratum, results or verdicts, and with the data facts in `review_prompt.json` (two sample rules that select the same 391 stores with the same values are the same restriction). The correction uses these labels; unresolved pairs (7 tail, 8 body) are excluded. |
| review_reason | The reviewer's one-sentence reason. |
| review_session | Opaque id of the reviewer session that labeled the pair (one session per batch). |
| first_review_status | `classified` or `unresolved` in the first declaration review (the same comparison without the data facts, 17 October 2026), whose labels are superseded. Its 2,989 classified replays are those the soundness review covered and the descriptive displays use. |

## review_prompt.json

The material every fact-review pair carried besides the candidate declaration: the comparison instruction, the
data-facts paragraph, the original's declaration (the target) and the public context (`DESIGN.md`, `CODEBOOK.md`).
The reviewer's session wrapper is `code/09_card_agents/data_generation/terra/rereview_gl.py` (arm `luna-fact`).

## choice-codes.csv

The coded final choices of each replay's declaration (3,000 rows), from GPT-6.1-Sol helper sessions working on
blinded packets (rule-based candidates, reviewed against each declaration's text; no strata, results or verdicts), 18
October 2026. The coding rules are in `code/08_card/data_generation/descriptive_coding/` (`choice-coding-schema.md`,
`choice-coding-clarifications.md`); this file holds the schema's display codes.

| Field | Values |
|---|---|
| session_id | Replay id. |
| measure | Employment measure: `nonmanager_fte` (full-time plus half of part-time, excluding managers), `full_time_employees` (the full-time count alone), `employee_headcount` (full-time plus part-time employees counted equally, with or without managers), `manager_inclusive_fte` (the FTE measure plus managers), `other_unclear`. |
| sample_rule | `complete_case_no_status_filter`, `closure_recoding` (closed stores' wave-2 employment set to zero), `respondents_only` (stores interviewed in wave 2), `other_unclear`. |
| specification | `nj_only` (New Jersey indicator only), `nj_additive_chain` (plus chain indicators), `added_baseline_covariates`, `other_unclear`. |

## replay-programs.jsonl.gz

Each replay's final `analysis.R` (`analysis_R`), rebuilt by `data_generation/rebuild_programs.py` from the tool calls
in `replay-requests/`, with `status`: `ok` (2,940 replays), `patches_skipped` (54; a patch did not apply and was
skipped whole) or `no_analysis_R` (6 replays wrote it in a way the record does not show; `analysis_R` is null). The
rebuild script also defines `compacted` (the conversation was compacted); no replay has it.

## replay-results.csv

Each recovered program run on its replay's subset (written and read back as the replay's `data.csv` was), in a
sandbox with no network (`data_generation/replay_reruns/`), 29 October 2026: `estimate`, `se`, `p_value` (17
significant digits), `error` (`NA`, or the error message; 2,994 of 3,000 return a result), and `estimate_printed`,
`p_value_printed` (`True` when the value equals a number the replay printed in its session at the printed precision).

## soundness-review.csv.gz

The methodological review behind Table 7, for the 2,989 replays classified by the first declaration review. The
reviewer saw the design, the codebook, the replay's declaration and its final `analysis.R` with comments removed, and
no results (`prompts/soundness-review-prompt.txt`); the auditor saw the same material, facts about the data and the
reviewer's verdict and reason (`prompts/soundness-audit-prompt.txt`). For the 6 replays without a recovered program
the reviewer saw the declaration alone and passed all six; the table and the audit leave them out.

| Field | Meaning |
|---|---|
| session_id | Replay id. |
| review_model | `gpt-6.1-sol` (high effort, Codex CLI, ChatGPT Pro subscription login, 29 October 2026). |
| review_pass | `true` when the reviewer found no major methodological error. |
| review_reason | Its reason. |
| audit_model | `claude-opus-5-5` (high effort, Claude Code CLI 2.1.314, Max subscription, 30 October 2026); empty for the 6 replays without a recovered program. |
| audit_major_error | `true` when the auditor found a major error by its stated standard. |
| audit_reviewer_correct | `true` when the auditor agreed with the reviewer's verdict. |
| audit_reason | Its reason. |

## activity-steps.csv.gz

The steps of each replay behind Figure 7: its shell commands, each with the main purpose an annotator agent coded for
it, and its file edits, in event order. Purposes come from a two-pass annotation of every replay's recorded analysis
by gpt-6-luna (a first pass at low effort and a full-source audit pass at high effort, through the Codex CLI on a
ChatGPT login and then on the OpenAI API), with
gpt-6-sol (high effort; the model id as recorded, a different model from GPT-6.1-Sol) checking a fixed sample of 96
replays, 18 to 20 October 2026. Replays with no usable annotation (16) have no rows. The annotators' command codebook
is `code/08_card/data_generation/descriptive_coding/step-annotation-contract.md`; `purpose` is its `main_purpose`.

| Field | Meaning |
|---|---|
| session_id | Replay id. |
| step | Position in the replay's sequence (1, 2, ...). |
| kind | `command` or `file_edit`. |
| purpose | For a command, the coded main purpose: `setup`, `inspect`, `construct`, `evaluate`, `interpret`, `choose`, `revise`, `verify`, `report`, `debug`, `mixed`, `other` or `unknown`. Empty for a file edit. |
| paths | For a file edit, the paths it changed inside the session's container, separated by `;`. Empty for a command. |

## analysis-streams.jsonl.xz

The full event stream of the analysis turn of every session (3,001 lines: the original and the 3,000 replays; xz
compressed), as Codex emitted it: messages, shell commands with their output, and file edits (paths only). Fields:
`case_id` (the replay's session id above; `original-001` for the original), `coding_id` (the opaque id the annotation
and coding used; null for the original), `is_original`, `schema`, `eventstream_bytes` and `eventstream_sha256` (of
the stream before redaction) and `analysis_events_jsonl` (the stream, one JSON event per line).

Redaction (`code/08_card/data_generation/redact.py`): every path in the cluster's software tree is replaced by
`<cluster-software-path>`, every line naming one of its administrative folders by `<redacted host folder>`, and the
owner and group fields of every `ls -l` line, which showed the numeric ids of the account that ran the sessions, by
`<owner>` and `<group>`. Nothing else changed; streams without such text keep their exact bytes.

## replay-requests/

For each replay, the tool calls (with their full input, including file patches) and the user messages (first 300
characters) of its final model request, which carries the whole conversation; four files of 750 replays. Fields:
`case_id` (the runtime's internal case key), `job_id` (the session id above), `request_id` (the runtime ledger's id of
that final request, 32 hex characters), `items` (`t`: `custom_tool_call` with `name` and `input`, or `user` with
`text`; the extractor also defines `function_call` and `compaction`, which occur in no record). Extracted read-only on
the cluster from the replays' closed case packs by `data_generation/replay_reruns/gl_extract_calls.py`, with the
streams' redaction applied (the extracts hold no command output, so only the software-tree rule changes anything).

## prompts/

`developer-instructions.txt` (the operator's instruction in every session's configuration),
`soundness-review-prompt.txt` and `soundness-audit-prompt.txt` (templates; `{{DESIGN}}`, `{{CODEBOOK}}`,
`{{DECLARATION}}`, `{{CODE}}`, `{{VERDICT}}`, `{{REASON}}` are filled per replay by the runners in
`code/08_card/data_generation/soundness/`).
