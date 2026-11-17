# card_sonnet: Card-Krueger replays with Claude Sonnet 5.5

One cohort, never pooled with another. All files are author-generated records of agent sessions.

## replays.csv.gz

One row per rostered replay, in roster order (gzip-compressed CSV, UTF-8).

| Field | Meaning |
|---|---|
| session_id | Opaque case id of the replay (20 hex characters); the same id names the case in the review files. |
| cohort | `card_sonnet`. |
| role | `replay`. |
| model | Model recorded in the session (`|`-separated if several; the planned model for a replay that did not run). |
| effort | Reasoning effort recorded in the session. |
| route | `claude_subscription`: a subscription login, not the metered API. |
| account | Which of the two logins ran the session (`a`). |
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

## review_repeats.csv

Pairs the reviewer read a second time in another batch, to measure its consistency: session_id, label, reason.

## review_prompt.json

The material each review pair carried besides the two declarations (identical in every batch). Keys: `instruction` (the comparison instruction), `data_facts` (the paragraph on the two equivalent sample
rules in these data; in the masked cohort with the masked column names), `target_declaration` (the original's
declaration, the review target, with keys model, null, population, outcome, sample_restrictions) and
`public_context` (a list of the agents' `DESIGN.md` and `CODEBOOK.md`, each with keys name, sha256, text). The
reviewer's session wrapper is `code/09_card_agents/data_generation/terra/rereview_gl.py`.

## original.json

The cohort's original run on all 410 stores: session id, model, effort, route, `p_value` (u, the full-data p-value
of its emitted program, the threshold between tail and body), `estimate`, `commitment` (its declaration, the
review target), and `final_report` (its last message; sandbox paths shortened to the workspace).

## session_flags.csv

One row per session (the original and 3,000 replays), extracted from the session archives: session_id (`original`
for the original), stratum (`tail`, `body`, or `original` for the original), decline_thinking,
decline_messages, decline_report (patterns of the decline rule, `code/09_card_agents/declines.py`, matching the
visible thinking, all messages and the final report; `|`-separated), and name_thinking, name_messages,
name_tool_calls (commands and file contents written), name_report ("krueger" found, `True`/`False`). A replay
declines when its thinking or messages match, and names the study when any of thinking, messages or tool calls
does.

## programs/original-analysis.R, original/

The original's emitted program (the fixed p-value procedure behind the bank,
`data/intermediate/09_card_agents/sonnet_bank_p.csv`), its written report, and its record of the specifications it ran,
`specification_log.csv` (columns id, spec, estimate, se, p_value, n; 33 specifications, M the main one), as written by the agent.

## decline_flags.csv, naming_counts.json

The decline flags and naming counts the cluster scans gave; `04_keyword_rates.py` checks `session_flags.csv` against
them. decline_flags.csv: session_id, stratum, thinking, messages, report (the
same patterns as session_flags.csv's decline columns) and excerpt (the first match with 60 characters before and
40 after, taken from the final report, else the messages, else the thinking; empty when nothing matches).
naming_counts.json keys: `replays`, `anywhere` (thinking, messages or tool calls), `in_tool_calls`,
`in_visible_thinking`, `in_messages`, `in_final_report`; all count replays of the analysis turn.
