# card_sol: Card-Krueger replays with GPT-6.1-Sol

One cohort, never pooled with another. All files are author-generated records of agent sessions.
GPT-6.1-Sol at high reasoning effort through Codex CLI 0.165.0 on the cluster, on two ChatGPT Pro logins, run from
29 October to 3 November 2026 and reviewed from 31 October to 3 November. Each replay ran in its own folder with its
own `/tmp`, inside an outer sandbox that hid the cluster's shared file systems.

## replays.csv.gz

One row per rostered replay, in roster order (gzip-compressed CSV, UTF-8).

| Field | Meaning |
|---|---|
| session_id | Opaque case id of the replay (20 hex characters); the same id names the case in the review files. |
| cohort | `card_sol`. |
| role | `replay`. |
| model | Model recorded in the session (`|`-separated if several; the planned model for a replay that did not run). |
| effort | Reasoning effort recorded in the session. |
| route | `chatgpt_subscription`: a subscription login, not the metered API. Empty for the 10 `unavailable` replays. |
| account | Which of the two logins ran the session (`primary`, `second`). Empty for the 10 `unavailable` replays. |
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

## programs/original-analysis.R

The original's emitted program, `analysis(data)` returning estimate, se and p_value: the fixed p-value procedure
run on every reference subset (`data/intermediate/09_card_agents/sol_bank_p.csv`, `code/09_card_agents/02_score_bank.R`).

## original/report.md, original/all_analyses.csv

The original's written report and its record of the 26 analyses it ran (T01 to T26), as written by the agent;
the selected analysis is T08 (Holm p = 0.700). Columns of all_analyses.csv: outcome (fte, worker_fte, headcount,
full_time, part_time, symmetric_fte), adjusted (chain and ownership controls), closure (permanent_zero,
all_closed_zero, respondents_only), test_id, estimate, se, p_value (nominal), ci_low, ci_high (95%), n, n_nj,
n_pa, df (residual degrees of freedom), p_holm (Holm over the 26), selected (the analysis it chose),
bonf_ci_low, bonf_ci_high (Bonferroni-adjusted interval).

## report_reads.csv

Blind reads, by GPT-6.1-Sol at high effort, of the final reports of the first 831 completed replays, with statistics
extracted as written; 40 reports were read twice (`read` = 2). Fields: session_id, read, nominal_p (the lead finding's unadjusted p), tests_stated (number of tests
the report says it ran and corrects for), other_significant, own_correction (HOLM, BONFERRONI, OTHER or NONE),
correction_tests, adjusted_p, survives (whether the report says the lead finding survives its own correction),
quote (the report's sentence stating the correction). Read by module 10.

## reports.jsonl.gz

The final report of each of the 2,990 completed replays: the last agent message of its analysis turn, from
the session archive of the replay's final run (gzip-compressed JSON lines, sorted by session_id). Fields:
session_id, stratum, final_report (verbatim; sandbox paths shortened to the session's workspace). These are
the texts the blind reads in `report_reads.csv` read. No script reads this file.

## session_flags.csv

Keyword flags for each of the 3,001 session archives (the 2,990 final runs, the 10 sessions cut by provider
errors and one earlier run that was rerun). Fields: session_id; archive (the archive's name, the case id and a
millisecond start time); final (`True` for the run whose records count); stratum; reasoning_summary_text (any
reasoning-summary text saved; none is); decline_messages and decline_reasoning_summaries (patterns of the decline
rule, `code/09_card_agents/declines.py`, `|`-separated); name_messages, name_reasoning_summaries, name_tool_calls
(commands and patches the agent wrote), name_tool_output ("krueger" found, `True`/`False`). The flags read the
session logs of all three turns, including command output.

## keyword_counts.json

The counts the cluster scans gave. Keys: `sessions_scanned` (3,001 session archives), `naming_krueger` (sessions
naming Krueger anywhere), `declines_in_messages_or_reasoning_summaries`, `with_reasoning_summary_text`;
`04_keyword_rates.py` checks `session_flags.csv` against them.
