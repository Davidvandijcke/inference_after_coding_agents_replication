# sol_common_analysis

GPT-6.1-Sol (high effort, Codex CLI 0.165, ChatGPT Pro logins, no API) on the open-ended simulation design with
the no-explore prompt (`code/10_self_correction/data_generation/sol_common_analysis/frozen/research_prompt.txt`:
one compelling, defensible finding, without the instruction to explore). One session per dataset, no replays.
Built by `code/10_self_correction/data_generation/extract_records.py`. The datasets are in
`data/raw/pathways_sol_common/`.

- **Planted** (`planted_sessions.csv`, 60): datasets with a moderate effect on employment at 12 months (latent
  shift 0.28 in `frozen/generate.py`), 30 in cohort `twin-panel-sol` (seeds 9,990,000 + slot, primary login) and 30
  in `twin-panel2-sol` (seeds 9,990,030 + slot, second login), 1 November 2026. The second 30 were added after the
  first cohort's originals had been read and before any conditional-correction verdict existed.
- **Reference** (`reference_sessions.csv`, 100, cohort `reference-null`): fresh null datasets (seeds 12,020,000 +
  slot, primary login), 2 November 2026, for the conditional correction of the common analysis.

## Shared fields

| Field | Meaning |
|---|---|
| `session_id` | opaque case id; also names the program in `programs/` |
| `cohort`, `role`, `model`, `effort`, `harness`, `route`, `account` | cohort (`twin-panel-sol`, `twin-panel2-sol` or `reference-null`); `original`; `gpt-6.1-sol`; `high`; Codex CLI 0.165; `chatgpt_subscription`; login (`primary` or `second`) |
| `slot`, `seed`, `tau` | dataset index (0 to 29 in each planted cohort, 0 to 99 for the reference sessions), generator seed and planted latent shift |
| `data_file`, `data_sha256` | dataset in `data/raw/pathways_sol_common/`, with the slot written in three digits (`panel-007.csv.gz`, `panel2-007.csv.gz`, `reference-007.csv.gz`), and the SHA-256 of the uncompressed file as served |
| `status` | `completed` for all sessions |
| `u` | p-value the emitted program returns on the session's dataset |
| `commit_model`, `commit_null`, `commit_population`, `commit_outcome`, `commit_sample_restrictions` | the session's declaration |

## planted_sessions.csv only

| Field | Meaning |
|---|---|
| `outcome`, `followup`, `restricted`, `interaction`, `group`, `baseline_columns` | the selected analysis, coded from program and declaration by the choice coder `data_generation/four_agent_null_run/code_choices.py` |
| `robust_se`, `reference_distribution` | the program's standard errors and test reference, read from its code by the procedure audit's rule (`procedure` in `extract_records.py`) |
| `reviewer`, `tests_stated`, `other_significant`, `own_correction`, `correction_tests`, `adjusted_p`, `survives`, `read_quote` | blind read of the final report by GPT-6.1-Sol under the facts rule of the four-agent null run (`four_agent_null_run/review2_rule_facts.txt`), from the raw reader output joined through the review's private key |

## reference_sessions.csv only

| Field | Meaning |
|---|---|
| `common_procedure` | `True` when program and declaration select employment at 12 months, all participants, treatment main effect with site indicators only, HC2 errors and a Student-t test (`reference_runs/collect_and_score.py`) |
| `reader_label`, `reader_reason` | for those sessions, a blind GPT-6.1-Sol reader's verdict (`MATCH`, `NONMATCH`, `AMBIGUOUS`) on whether the declaration is that analysis without a stronger model assumption (`reference_runs/READER_PROMPT_FROZEN.txt`); the reader saw the declaration and facts coded from the program |

## reports.jsonl.gz and programs/

`session_id`, `final_report` for all 160 sessions; `programs/<session_id>.py` is each session's `score.py`.

## Not shipped

Transcripts and workspaces stay in the project's cluster folders `multiplicity-comparison-20261031/twin-panel-sol/`
and `twin-panel2-sol/` (planted) and `sol-recurrence-power-20261102/cohorts/reference/` (reference). The 200 earlier
no-explore null sessions (cohort `twin-nulls-sol`) and the 160 replays of two planted originals are not shipped.
