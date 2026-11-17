# 02 Open-ended agent design

Table 1 Panel B and Figure 3. An agent analyzes a synthetic randomized evaluation in which every treatment effect is
zero and reports a p-value; replay agents repeat the task on subsamples of 200 of its 2,000 participants.

## Analysis

`run.sh` takes about 2 minutes.

| Script | Writes |
|---|---|
| `01_replay_counts.py` | `data/derived/02_open_ended/replay_counts.csv`: replays and matches per original |
| `02_replay_intervals.R` | `per_original.csv`: corrected p-value, multiplier and intervals per original |
| `03_panel_b_figure.py` | `paper/tabs/simulation_panel_b.tex`, `paper/figs/simulation_overview.pdf`, `results/numbers/02_open_ended.json` |

Inputs: the 100 datasets in `data/raw/pathways/`, the session records in `data/agent_runs/open_ended_luna/` and each
original program's p-values on its subsamples in `data/intermediate/02_open_ended/`. Figure 3(a) also reads the
derived outputs of modules 01 and 03. A match is the replay agent's own verdict.

## Code that produced the agent records (`data_generation/`, not rerun)

1. `generate.py`, `cluster/make_originals.py`: the 100 datasets, in which every treatment effect is zero.
2. `cluster/runner.py` with `prompts/`: one sandboxed session per dataset (analysis, then commitment). The same
   script runs the replays, which end with the comparison prompt.
3. `cluster/make_references.py`: scores each original's program on subsamples of 200 and draws the tail and body
   replays. `cluster/extend_references.py` draws the second batch.
4. `extract_records.py`: wrote the shipped records.

The agent is GPT-6-Luna at high reasoning effort through the Codex CLI; the cohort is described in
`data/agent_runs/open_ended_luna/CODEBOOK.md`. Rerunning the sessions needs model access and gives a new cohort,
since agents sample their outputs.
