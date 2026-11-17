# 08 Card-Krueger application

Figures 5 to 8 and Tables 6 and 7. GPT-6-Luna analyzes Card and Krueger's (1994) survey of 410 fast-food restaurants
once on all the data; replays of the same agent repeat the task on 3,000 subsets of 150 restaurants drawn from a bank
of 20,000.

## Analysis (run by `run_all.sh`, about 1 minute)

| Script | Writes |
|---|---|
| `01_original.R` | `data/derived/08_card/original.csv`: the original program on all stores, and Card and Krueger's Table 4 (ii); checks the agents' `data.csv` against `public.dat` |
| `02_bank.R` | `bank.csv`; redraws the 20,000 subsets and the replay roster from their seeds and reruns the original program on every subset |
| `03_correction.R` | `correction.csv`: matches, corrected p-value, multiplier and one-sided 95% bounds |
| `04_threshold.py` | `threshold.csv`: the multiplier at the 5% level and its lower bound |
| `05_selection_figure.py` | `paper/figs/card_selection.pdf` (Figure 5) |
| `06_choice_pvalues.R` | `choice-pvalues.csv`: each replay's subset scored under other measures, specifications and sample rules |
| `07_choice_figure.py` | `paper/figs/card_choice_significance.pdf` (Figure 6), `choice-summary.json` |
| `08_replay_figure.py` | `paper/figs/card_replay_significance.pdf` (Figure 8), `paper/tabs/card_replay_review.tex` (Table 7), `replay-summary.json` |
| `09_activity.py` | `paper/figs/card_activity_timeline.pdf` (Figure 7), `activity.json` |
| `10_variables_table.py` | `paper/tabs/card_variables.tex` (Table 6) |
| `11_numbers.py` | `results/numbers/08_card.json` |

Inputs: `data/raw/card_krueger/` (Card and Krueger's files, redistributed unmodified with attribution; see its
`SOURCE.md`), `data/agent_runs/card_luna/` (session records, see its `CODEBOOK.md`) and `data/intermediate/08_card/`
(the reference bank). The correction uses the 2,985 replays with a match label. Figure 6 uses the 2,989 replays that
an earlier comparison of the declarations, made without the data facts, classified; Figure 8 and Table 7 the 2,983 of
them with a final program that runs; Figure 7 the 2,944 with a purpose code for every command. Figure 8 weights tail
and body replays to the bank; the other displays pool them.

## Code that produced the agent records (`data_generation/`, not rerun)

1. `agent_runtime/`: `shared/studies/codex_cli/linux_runner.py` runs a session (Codex CLI 0.157.0) in the container
   of `linux_boundary.py` under `network_policy.py`, with the commitment and comparison prompts of `packet/` in its
   folder. The original ran `initialize_case` (`packet/` and the research prompt), `freeze_initial_state`
   (`replays/candidate/studies/codex_cli/initial_state.py`), `run_analysis` and `run_commitment`.
2. `build_bank.R [OUT]`: the 20,000 subsets, the original program's p-value on each and the 3,000 replay datasets.
3. Each replay: `clone_initial_state` (with its `data.csv`), `run_analysis`, `run_commitment`, `run_comparison`.
4. `first_review/review_pipeline.py`: `export` (blinded batches of up to 64 declaration pairs), `integrate` (labels).
5. Match labels: module 09's `terra/rereview_packets.py`, `rereview_gl.py` and `rereview_join.py`.
6. `descriptive_coding/choice/`: `extraction/extract.py` (cases), `schema/propose_codes.py` (candidates; categories in
   `SCHEMA.json`), the coders' `coding/A/rebuild.py`, `B/lock_and_rebuild.py` and `C/build_codes.py` (a third of the
   cases each), `integration/package_results.py` (join); rules in `descriptive_coding/choice-coding-*.md`.
7. `streams/inventory.py` (each session's place in the case packs), then `streams/extract.py` (the event streams).
8. `descriptive_coding/steps/`: `pipeline/` (cases, schemas, quote checks); `native/run_reviews.py` and
   `run_source_audits.py` (prompt and CLI call of the first and second pass); `reporting/closed_results.py` with
   `features.py` and `reviewed_choice_adapter.py` (join); rules in `descriptive_coding/step-annotation-contract.md`.
9. `replay_reruns/gl_extract_calls.py` (`python3 - PACK < gl_extract_calls.py`); `rebuild_programs.py`, whose output
   the rerun read with the id field named `replay_id`; `replay_reruns/run_replay_functions.R IN OUT` under
   `sandbox-exec -f sandbox.sb`.
10. `soundness/`: `make_packets.py`, `review.py` (imports module 09's `terra/pilot.py`); `build_cases.py OUT` (with
    `audit.py`), then `audit_gl.py` as 12 Slurm array tasks (`TASKS=12`; `RUN`, `CLAUDE_BIN`, `LOCAL_TMP` set);
    `audit_failures.R` and `trace_fallback.R`, the check of 21 failures (`failure-audit.csv`, `fallback-trace.txt`).
11. `extract_records.py RESEARCH/output`, with `rebuild_programs.py` and `redact.py`: the shipped records and bank.

Cohort: GPT-6-Luna (high effort, Codex CLI 0.157.0, OpenAI API), one container per session on the Great Lakes
cluster; the original ran on 16 October 2026 and the replays on 17 October (about $122 at token prices). The 1,142
tail and 1,858 body replays follow a Neyman allocation from the match rates of 743 earlier replays of the same
original, which serve only this purpose and are not shipped. The first review (the earlier comparison of the
declarations) was made by 24 GPT-6.1-Sol subagents (high effort) of a Codex Desktop session on 17 October. A match
is a label by GPT-6.1-Sol (high effort, Codex CLI 0.165.0, ChatGPT Pro logins, 30 October), which compared each
declaration with the original's in blinded batches of 60, given checked facts about the data (`review_prompt.json`);
its 15 unresolved pairs are excluded. Table 7's reviewer is GPT-6.1-Sol (high effort, Pro login, 29 October), which
read each declaration and comment-free final program; Claude Opus 5.5 (high effort, Claude Code CLI 2.1.314, Max
subscription, 30 October) audited its verdicts. "Reviewer + audit" keeps the reviewer's verdicts, except that 41 of
its 46 failures pass, which a check of the code and data found to be no error, and 9 of its passes fail, which
report a p-value adjusted for undeclared multiple testing, the error type the audit surfaced. The choice codes come
from GPT-6.1-Sol sessions on blinded declarations (18 October). Figure 7's purposes come from two passes by
gpt-6-luna (low, then high effort) through the Codex CLI, on a laptop with a ChatGPT login and for the remaining
cases through the OpenAI API on the cluster; gpt-6-sol (not GPT-6.1-Sol) checked 96 replays (18 to 20 October).

Not included: the billing relay through which sessions reached the API; the deployment records with each phase's
time and token limits; the coders' and reviewers' saved decisions; the messages that started the subagents (kept
only encrypted). Four steps were commands typed in sessions: building the annotation cases with `streams/extract.py`'s
`normalize()`, merging later checks into the choice codes (162 of 9,000 changed), adding a timestamp field to two
first-review label files, and recompressing the streams to `.xz`. In the archived code, accounts, paths and proxy
hosts are placeholders such as `<slurm-account>`. Earlier versions, job files and helper scripts of the runs are not
included and are available from the authors.
