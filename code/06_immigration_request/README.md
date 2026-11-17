# 06 Realistic request in the immigration application

Figure 4 and Table 3. A coding agent writes an evidence brief for a team that argues for open immigration and
reports a p-value; replays use subsamples of 16 of the 46 countries, and 60 more sessions give Miao et al.'s m-value.

## Analysis

`run.sh` takes about 25 seconds.

| Script | Writes |
|---|---|
| `01_correction.R` | `data/derived/06_immigration_request/correction.csv`: matches, corrected p-value, multiplier and intervals |
| `02_figure.py` | `paper/figs/issp_selection.pdf` |
| `03_mvalue.py` | `mvalue.json`: the m-value, the reference sessions' leads and the clustering of their plans |
| `04_variables_table.py` | `paper/tabs/issp_variables.tex` |
| `05_numbers.py` | `results/numbers/06_immigration_request.json` |

Inputs: `data/agent_runs/issp_request_luna/`, `issp_mvalue_luna/` and `data/intermediate/06_immigration_request/`.

The match counts behind Figure 4, 79 of 500 tail and 32 of 500 body replays, are the replays' own verdicts (31 and 7)
plus 73 of the 962 nonmatches that a second reading counted as matches because their stated differences were wording
only. A Claude Code session (Claude Opus 5.5) did that reading on 28 October 2026: it read the stated reasons of 90
nonmatches (86 picked by keyword rules and 4 others), not blind to stratum, and left the rest unread.

## Code that produced the agent records (`data_generation/`, not rerun)

1. Module 02's `cluster/runner.py`, adapted to whole-country data files, ran every session on the ISSP file that
   `code/05_immigration_personas/data_generation/build_issp_file.py` builds.
2. `agents/make_replays.py`: scores the original's program on 4,000 subsamples and draws the tail and body replays.
3. `agents/reference/make_cases.py`: the 60 reference sessions, which stop after the analysis.
4. `extract_request_records.py` wrote the cohort folders and `score_bank.py` the bank's p-values and full-file output.

The agent is GPT-6-Luna at high effort; the cohorts are described in their `CODEBOOK.md`. Reruns need model access.
