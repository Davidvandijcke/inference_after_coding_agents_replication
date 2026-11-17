# 03 Real-data placebo design

Table 1 Panel C and the real-data bar of Figure 3(a). The open-ended design of module 02 runs on the data of three
published experiments with a placebo treatment, so every null about the treatment is true; replays use 10% subsamples.

## Analysis

`run.sh` takes about 90 seconds.

| Script | Writes |
|---|---|
| `01_build_base_files.R` | `data/derived/03_placebo/base/`: one base file per study, without treatment, and `experiments.json` |
| `02_make_placebo_datasets.py` | `datasets/original_NNN.csv`: the 102 datasets, checked with every replay subsample against the data the agents received |
| `03_replay_counts.py` | `replay_counts.csv`: strata, replays and matches per original |
| `04_replay_correction.R` | `per_original.csv` (read by module 02) and `replay_intervals.csv`: corrected p-value, multiplier and intervals per original |
| `05_identity_check.py` | `identity_check.json`: the session records with words that name a source study |
| `06_panel_c.py` | `paper/tabs/simulation_panel_c.tex`, `results/numbers/03_placebo.json` |

Inputs: the three studies' data in `data/raw/`, the session records in `data/agent_runs/placebo_luna/` and each
program's p-values on its subsamples in `data/intermediate/03_placebo/`. A match is the replay agent's own verdict.

## Code that produced the agent records (`data_generation/`, not rerun)

1. `build_teachers_base.R`: the teachers study's extract in `data/raw/`, from the authors' deposit.
2. `runtime/generate.py` and a version of module 02's `cluster/make_originals.py` adapted to it: the 102 datasets.
3. A version of module 02's `cluster/runner.py` adapted to ChatGPT logins: the sessions and replays, with
   `runtime/research_prompt.txt`, `runtime/codebooks/<study>.md` and module 02's commitment and comparison prompts.
4. A version of module 02's `cluster/make_references.py` adapted to each study's size: the banks and replay draws.
5. `export_cluster_records.py`: wrote the shipped records and each program's p-values on its subsamples.

The agent is GPT-6-Luna at high reasoning effort through the Codex CLI; the cohort is described in
`data/agent_runs/placebo_luna/CODEBOOK.md`. Rerunning the sessions needs model access and gives a new cohort.
