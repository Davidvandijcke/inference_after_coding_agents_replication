# 08 Card-Krueger application

Figures 5 to 8 and Tables 6 and 7. An agent analyzes Card and Krueger's (1994) survey of 410 fast-food restaurants;
replays of the same agent repeat the task on 3,000 subsets of 150 restaurants drawn from a bank of 20,000.

## Analysis

`run.sh` takes about 1 minute.

| Script | Writes |
|---|---|
| `01_original.R` | `data/derived/08_card/original.csv`: the original program on all stores and Card and Krueger's Table 4 (ii) |
| `02_bank.R` | `bank.csv`: checks of the bank and the replay roster against a fresh draw and rescoring |
| `03_correction.R` | `correction.csv`: matches, corrected p-value, multiplier and one-sided 95% bounds |
| `04_threshold.py` | `threshold.csv`: the multiplier at the 5% level and its lower bound |
| `05_selection_figure.py` | `paper/figs/card_selection.pdf` (Figure 5) |
| `06_choice_pvalues.R` | `choice-pvalues.csv`: each replay's subset scored under other measures, specifications and sample rules |
| `07_choice_figure.py` | `paper/figs/card_choice_significance.pdf` (Figure 6), `choice-summary.json` |
| `08_replay_figure.py` | `paper/figs/card_replay_significance.pdf` (Figure 8), `paper/tabs/card_replay_review.tex` (Table 7), `replay-summary.json` |
| `09_activity.py` | `paper/figs/card_activity_timeline.pdf` (Figure 7), `activity.json` |
| `10_variables_table.py` | `paper/tabs/card_variables.tex` (Table 6) |
| `11_numbers.py` | `results/numbers/08_card.json` |

Inputs: `data/raw/card_krueger/`, the session records in `data/agent_runs/card_luna/` and the reference bank in
`data/intermediate/08_card/`. A match is a reviewer agent's label, given checked facts about the data.

## Code that produced the agent records (`data_generation/`, not rerun)

1. The sessions ran in a version of `code/02_open_ended/data_generation/cluster/runner.py` in which each replay starts
   from a copy of the original's initial workspace with its own `data.csv`.
2. `build_bank.R`: the 20,000 subsets, the original program's p-value on each and the 3,000 replay datasets.
3. `first_review/review_pipeline.py`: the first review's blinded pairs. Matches: module 09's `terra/rereview_*.py`.
4. `descriptive_coding/`: the coders' rules (`choice-coding-*.md`; Figure 6); the annotators' codebook
   (`step-annotation-contract.md`) and two passes (`steps/native/run_reviews.py`, `run_source_audits.py`; Figure 7).
5. Table 7: `soundness/review.py` (review), `audit_gl.py` (audit) and the check of 21 failures (`failure-audit.csv`,
   `fallback-trace.txt`). Figure 8: `replay_reruns/run_replay_functions.R` reran each replay's final program.
6. `extract_records.py` wrote the shipped records; `rebuild_programs.py` rebuilt each replay's final program.

The agent is GPT-6-Luna at high reasoning effort; the cohort is described in `data/agent_runs/card_luna/CODEBOOK.md`.
Rerunning the sessions needs model access and gives a new cohort, since agents sample their outputs.
