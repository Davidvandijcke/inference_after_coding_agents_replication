# 04 Power design

Table 1 Panel D. The open-ended design of module 02 with a true program effect on employment at 12 months (+6.2
and +9.6 percentage points, 100 datasets each); replays use subsamples of 200 of the 2,000 participants.

## Analysis

`run.sh` takes about 1.5 minutes.

| Script | Writes |
|---|---|
| `01_prespecified.py` | `data/derived/04_power/datasets.csv`: the pre-specified test's p-value and the realized Cohen's d per dataset |
| `02_replay_counts.py` | `replay_counts.csv`: u, replays and matches per usable original; `program_benchmark.csv`: each program's rejection rate on new data |
| `03_replay_bounds.R` | `bounds.csv`: the replay bounds per original |
| `04_panel_d.py` | `paper/tabs/simulation_panel_d.tex`, `results/numbers/04_power.json` |

Inputs: the 200 datasets in `data/raw/pathways_power/`, the session records, lead codes and programs in
`data/agent_runs/power_luna_login/`, and each program's p-values on its subsamples and on new datasets, with the
effect calibration, in `data/intermediate/04_power/`. A match is the replay agent's own verdict.

## Code that produced the agent records (`data_generation/`, not rerun)

1. `generate.py`, `calibrate_effect.py`: module 02's generator with the effect added, and the effect sizes.
2. Module 02's `cluster/make_originals.py`, then a version of its `cluster/runner.py` adapted to ChatGPT logins, with
   its `prompts/`: the 200 datasets and one session per dataset. The same script runs the replays.
3. `code_headlines.py`: the lead codes, from the programs and commitments, before any bank.
4. `bank_chunks.py` with `fast_guest.py` and module 02's `cluster/make_references.py` (the same file): the banks,
   each program's p-values on new datasets and the draw of the replayed subsamples.
5. `extract_records.py`: wrote the shipped records.

The agent is GPT-6-Luna at high reasoning effort through the Codex CLI; the cohort is described in
`data/agent_runs/power_luna_login/CODEBOOK.md`. Rerunning the sessions needs model access and gives a new cohort.
