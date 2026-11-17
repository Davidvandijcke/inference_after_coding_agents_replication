# card_placebo_sol

GPT-6.1-Sol (high effort, Codex CLI 0.165, ChatGPT Pro logins, no API) analyzing 200 full-size placebo versions of
the Card-Krueger data (`data/raw/card_placebo/`), one session per dataset, no replays, 2 November 2026: a pilot of 40
datasets and then 160 more (`stage`). The prompt
(`code/10_self_correction/data_generation/card_placebo/frozen/research_prompt.txt`) is the Card prompt plus a
program-output paragraph: `p_value` must be the unadjusted p-value of the chosen main test, and a
multiplicity-adjusted p-value goes in a separate field and is explained in the report. The codebook differs from the
Card codebook only in describing the store identifiers as anonymous, and the packet also held the Card study
description `data/agent_runs/card_luna/packet/DESIGN.md`. Built by
`code/10_self_correction/data_generation/extract_records.py`; the inputs are in
`code/10_self_correction/data_generation/card_placebo/`.

## sessions.csv (200 rows)

| Field | Meaning |
|---|---|
| `session_id` | opaque case id |
| `cohort`, `role`, `model`, `effort`, `harness`, `route` | `card_placebo_sol`, `original`, `gpt-6.1-sol`, `high`, Codex CLI 0.165, `chatgpt_subscription` |
| `account` | which of two Pro logins ran the session (`primary`, `second`) |
| `stage` | `pilot` (the 40 datasets run first; their slots are spread over the roster, from 3 to 196) or `confirmation` (the other 160) |
| `slot`, `seed` | dataset index (0 to 199) and generator seed (117,020,000 + slot); the dataset is `data/raw/card_placebo/placebo-<slot>.csv.gz`, with the slot written in three digits (`placebo-007.csv.gz`) |
| `data_sha256` | SHA-256 of the dataset as served, from the runner record |
| `status` | `completed` for all 200 |
| `u` | p-value the emitted program `analysis.R` returns on the session's dataset, unadjusted by the prompt's output contract |
| `selected_class` | class of the declared analysis: `workerFTE_allframe` (worker FTE on the full store frame, 11 sessions), `other_employment_allframe` (another employment measure on the full store frame, 141), `restricted_or_heterogeneity` (a restricted population or a heterogeneity test, 47), `outside_known_null:WAGE_BASED` (a wage-based exposure, which the permutation does not make null, 1) |
| `commit_model`, `commit_null`, `commit_population`, `commit_outcome`, `commit_sample_restrictions` | the session's declaration of its selected model and null |

## reports.jsonl.gz

`session_id`, `final_report` (the agent's final message) for all 200 sessions.

## blind_reads.csv.gz (200 rows)

GPT-6.1-Sol (high effort) read each session's final message, saved report and declaration under
`code/10_self_correction/data_generation/card_placebo/reader_prompt.txt` (schema `reader_schema.json`). One row per
report, from the raw reader outputs joined to their sessions through each wave's private key. The prompt is a second
version: the 40 pilot reports were read again under it after an audit of their first reads, and the rows here are
those second reads; the first reads and the earlier prompt are not shipped. `case_id` is the session's `session_id`
and `slot` its dataset; `reader_cohort` is `pilot` or `confirmation`; `reader_wave` is the wave within it (four
pilot waves, five confirmation waves); `reviewer` is the reader model. Label fields follow the prompt:
`nominal_p`, `returned_p`, `returned_p_kind`, `outcome_construction`, `target_exposure`, `target_population`,
`status_missing_rules`, `restricted_population`, `heterogeneity_null`, `controls`, `scientific_model`, `exact_null`,
`tests_stated` (largest stated number of employment analyses searched), `own_correction`, `correction_tests`,
`adjusted_p`, `survives`, `lead_decision`, `alternative_headline_present`, `alternative_headline_summary`,
`alternative_nominal_p`, `quotes` (JSON).

## Not shipped

Workspaces, saved reports and transcripts stay in the project's cluster folder `card-conditional-size-20261102/`:
`lanes/sol-primary/node-archives/` and `lanes/sol-second/node-archives/` (pilot) and `extension/lanes/sol-primary/`
and `extension/lanes/sol-second/` (confirmation); the readers' session archives stay in the review folders.
GPT-6-Luna's 200 sessions on the same datasets, and their reads, are not shipped.
