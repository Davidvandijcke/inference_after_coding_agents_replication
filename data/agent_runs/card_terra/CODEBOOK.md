# card_terra: Card-Krueger replays with GPT-6.1-Terra

One cohort, never pooled with another. All files are author-generated records of agent sessions.

## replays.csv.gz

One row per rostered replay, in roster order (gzip-compressed CSV, UTF-8).

| Field | Meaning |
|---|---|
| session_id | Opaque case id of the replay (20 hex characters); the same id names the case in the review files. |
| cohort | `card_terra`. |
| role | `replay`. |
| model | Model recorded in the session (`|`-separated if several; the planned model for a replay that did not run). |
| effort | Reasoning effort recorded in the session. |
| route | `chatgpt_subscription`: a subscription login, not the metered API. |
| account | Which of the two logins ran the session (`primary`, `second`). |
| position | Position in the sealed run order (a random order, unrelated to stratum). |
| bank_id | Id of the reference subset (150 whole stores of 410) in the 20,000-subset Card bank of module 08 (`data/intermediate/08_card/`). |
| bank_p | p-value of the original's program on that subset. |
| stratum | `tail` when bank_p is at most the original's full-data p-value u (`original.json`, `p_value`), else `body`. |
| status | `completed`, or `unavailable` for a replay that did not finish (never retried; excluded). |
| commit_model, commit_null, commit_population, commit_outcome, commit_sample_restrictions | The replay's committed declaration of its selected analysis, verbatim. |
| own_match | The replay's own verdict, after seeing the original's declaration, on whether it selected the same model and null (`True`/`False`). |
| own_reason | The replay's reason for its verdict. |
| review_label | The reviewing agent's label (GPT-6.1-Sol, high effort, blind to stratum and verdict): `match`, `nonmatch` or `unresolved`. Table tab:card_terra uses these labels; unresolved pairs are excluded. |
| review_reason | The reviewer's one-sentence reason. |
| machine | Where the replay ran: `laptop` (368 replays, Codex CLI 0.164.0-alpha.11.1) or `cluster` (2,632 replays, Codex CLI 0.165.0). |
| read_isolation | `True` when the session ran under the sandbox profile that denies reads outside its own run folder; `False` for the first 263 laptop replays, which ran before that profile was added. |
| sandbox_refused | Number of the replay's shell commands, over its three turns, that the sandbox refused to run ("error building bubblewrap command"). Positive for 2,039 replays, all on the cluster. |
| outside_commands | Number of its shell commands whose text names a path outside its own run folder (project, home, storage or other runs' paths, `..`, a listing of `/` or `/tmp`), by the run's keyword rule. Positive for 319 replays. |
| tmp_exposed | `listed` (59 replays) when the replay listed, and `read` (10) when it read or ran, a file that another replay had left in a `/tmp` the two shared; `no` otherwise. |

The last five fields come from the run's audit of its environment (`processed/terra-replays.csv` and
`processed/tmp-exposure.json` in the run folder of the research repository; `machine` recoded to laptop and cluster).

## Environment defects of this cohort

The audit found two defects after the cohort had finished; the table keeps every classified replay.

1. Refused commands. On the cluster the sandbox scanned the shared work folder before each command, the scan
   sometimes failed while other replays changed files there, and the CLI then refused the command; agents mostly
   retried. The laptop replays had none.
2. Shared `/tmp`. On the laptop every replay shared one `/tmp`, and on the cluster the eight workers of an array
   task shared one. 285 replays used it; 69 listed or read a file another replay had left there, and 10 of them read
   or ran such a file. The audit found no replay that read the original's records, the roster, the bank p-values,
   another session's CLI home or a project file.

## review_prompt.json

The material each review pair carried besides the two declarations (identical in every batch). Keys: `instruction` (the comparison instruction), `data_facts` (the paragraph on the two equivalent sample
rules in these data; in the masked cohort with the masked column names), `target_declaration` (the original's
declaration, the review target, with keys model, null, population, outcome, sample_restrictions) and
`public_context` (a list of the agents' `DESIGN.md` and `CODEBOOK.md`, each with keys name, sha256, text). The
reviewer's session wrapper is in `code/09_card_agents/data_generation/terra/`. The Terra review read every pair
once, so this cohort has no `review_repeats.csv`.

## original.json

The cohort's original run on all 410 stores: session id, model, effort, route, `p_value` (u, the full-data p-value
of its emitted program, the threshold between tail and body), `estimate`, `commitment` (its declaration, the
review target), and `final_report` (its last message; sandbox paths shortened to the workspace).

## programs/original-analysis.R

The original's emitted program, `analysis(data)` returning estimate, se and p_value: the fixed p-value procedure
run on every reference subset (`data/intermediate/09_card_agents/terra_bank_p.csv`, `code/09_card_agents/02_score_bank.R`).

## original/REPORT.md

The original's written report from its workspace.
