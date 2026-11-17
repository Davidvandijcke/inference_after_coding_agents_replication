# 07 Known selection rules on a fixed menu

Table 5. Agents report one of 108 country-level OLS regressions of support for welfare items on the immigrant
share (the menu) by a fixed rule, for an analyst who expects a negative association and one who expects a positive
one; replay agents apply the same rule to subsets of 13 of the 46 countries.

## Analysis

`run.sh` takes about 10 seconds.

| Script | Writes |
|---|---|
| `01_exact.py` | `data/derived/07_immigration_menu/exact.csv`: each rule's exact corrected p-value over the 20,000 bank subsets |
| `02_replay.R` | `replay.csv`: match counts, corrected p-value, multiplier and bounds per arm and workflow |
| `03_table.py` | `paper/tabs/immigration_menu.tex`, `results/numbers/07_immigration_menu.json` |

Inputs: `data/agent_runs/issp_menu_anti/`, `issp_menu_pro/` and `data/intermediate/07_immigration_menu/`. A replay
is classified when it recorded a selection and matches when it selected the original's menu model.

## Code that produced the agent records (`data_generation/`, not rerun)

1. `prepare_anti.py`, `prepare_pro.py`: on the ISSP file that module 05's `build_issp_file.py` builds, each arm's
   20,000 bank subsets, each rule's pick, the exact values and the replay roster.
2. `agents/replay_anti.py`: the six originals, then the replays, each session with the files of its arm's `inputs/`;
   the pro arm ran a version adapted to its target, the sign its client-pressure rule expects and its prompts.
3. `extract_menu_records.py`: wrote both cohort folders from the run records.
4. `fit_menu_full_data.py`: `menu_full_data.csv`, the menu fitted on the full file with the agents' helper.

The agent is GPT-6-Luna at high effort; each arm is described in its `CODEBOOK.md`. Reruns need model access.
