# 07 Known selection rules on a fixed menu

Table 5. Agents report one of 108 country-level OLS regressions of support for welfare items on the immigrant
share (the menu) by a fixed rule, for an analyst who expects a negative association and one who expects a positive
one; replay agents apply the same rule to subsets of 13 of the 46 countries.

## Analysis (run by `run_all.sh`, about 10 seconds)

| Script | Writes |
|---|---|
| `01_exact.py` | `data/derived/07_immigration_menu/exact.csv`: each rule's exact corrected p-value over all 20,000 bank subsets |
| `02_replay.R` | `replay.csv`: match counts, corrected p-value, multiplier and bounds per arm and workflow; checks each replay's stratum against the bank |
| `03_table.py` | `paper/tabs/immigration_menu.tex`, `results/numbers/07_immigration_menu.json` |

Inputs: `data/agent_runs/issp_menu_anti/` and `issp_menu_pro/` (session records and the files every session
received, see their `CODEBOOK.md`) and `data/intermediate/07_immigration_menu/` (the bank subsets with each rule's
pick, and the menu fitted on the full file, see its `CODEBOOK.md`). The intermediate files are aggregates of the
ISSP analysis file, which is restricted and not shipped; `data/raw/issp/SOURCE.md` describes access.

## Code that produced the agent records (`data_generation/`, not rerun)

Each arm's run folder holds the session inputs in `inputs/` (from `data/agent_runs/issp_menu_anti/inputs/`, with the
pro arm's prompts from `issp_menu_pro/inputs/prompts/`, and the ISSP file as `data.csv`) and the scripts in `code/`:

1. `prepare_anti.py <issp_clean.csv> <run folder>`, `prepare_pro.py` (each as `prepare.py`): the 20,000 bank subsets
   (`default_rng(2026103021)`), each rule's pick, the exact values and the replay roster (seeds 2026103022 and
   2026103023 for anti, 2026103024 and 2026103025 for pro). About one minute each.
2. `agents/replay_anti.py`, `agents/replay_pro.py` (each as `code/replay.py`): `--originals --account
   primary=<login>:6` runs the six originals, then `--task <i> --tasks 10 --account primary=<login>:10` the replays.
   Apart from paths, the two differ only in the target model, the sign the client-pressure rule expects and the prompts.
3. `extract_menu_records.py <research repository>`: wrote both cohort folders from the run records.
4. `fit_menu_full_data.py <issp_clean.csv> <folder>`: `menu_full_data.csv`, the menu fitted on the full file with
   the agents' helper.

Cohort: GPT-6-Luna at high effort through Codex CLI 0.165.0 in a sandbox on a cluster, 30 October 2026, on a ChatGPT
Pro login, with no API spend. Every session received the menu (`CATALOGUE.json`), a helper that fits it with HC3
standard errors, an analysis contract and its workflow's prompts. The rules report the named specification whatever
its result (fixed model), the specification with the most negative or most positive t-statistic (frontier), or the
named specification if it has the expected sign and p < 0.05 and otherwise the most extreme alternative (client
pressure). Each arm has six full-data originals, two per workflow, all of which reported the target (anti: AD061,
u = 0.0256; pro: AD098, u = 0.7545, and AD099, u = 0.9189, for client pressure), and 1,000 replays (100 per stratum
for the fixed model, 200 for the others). A replay is classified when it recorded a selection and matches when it
selected the original's menu model; the replays' own verdicts are kept for reference. In the anti arm, 81 cases of
the first array failed (80 on rate limits) and reran from scratch before any match rate was looked at; one failed
again and is excluded. In the pro arm, four selections saved in a subfolder were recovered from the case archives
and count as classified, and one completed replay selected no model and is excluded. All 1,998 recorded selections
follow the assigned rule.

Not included: the ISSP analysis file (restricted) and the session transcripts (kept by the authors). The menu (built
by script for an earlier cohort on 16 October 2026) and the helper, the analysis contract and the prompts (written by
hand for that cohort) are shipped as the agents received them; the command that copied them and put AD061 and AD098
in place of the earlier targets in the prompts was run inline and never saved, and neither was the step that
recovered the four pro selections from the archives. Earlier versions, job files and helper scripts of the runs are
not included and are available from the authors.
