# 03 Real-data placebo design

Table 1 Panel C and the real-data bar of Figure 3(a). The open-ended agent design runs on the data of three
published randomized experiments with the treatment replaced by a placebo assignment, so every null about the
treatment is true: 34 datasets per study, each analyzed by an original agent and replayed on 40 tail and 200 body
subsamples at m/n = 0.10.

## Analysis (run by `run_all.sh`, about 90 seconds)

| Script | Writes |
|---|---|
| `01_build_base_files.R` | `data/derived/03_placebo/base/`: one base file per study (neutral names, randomization block, no treatment) and `experiments.json` |
| `02_make_placebo_datasets.py` | `datasets/original_NNN.csv`, the 102 datasets; checks them, the base files and all 24,440 replay subsamples against the SHA-256 recorded when the cohort ran |
| `03_replay_counts.py` | `replay_counts.csv`: bank strata, replays and own matches per original; checks each replay's stratum against the bank |
| `04_replay_correction.R` | `per_original.csv` (read by module 02), `replay_intervals.csv`: corrected p-value, intervals and multiplier per original |
| `05_identity_check.py` | `identity_check.json`: the search of all 24,542 session records for words that name a source study |
| `06_panel_c.py` | `paper/tabs/simulation_panel_c.tex`, `results/numbers/03_placebo.json` |

Inputs: `data/agent_runs/placebo_luna/` (see its `CODEBOOK.md`), `data/intermediate/03_placebo/bank_pvalues.csv.gz`
(each original program's p-value on every subsample of its reference bank) and the studies' data in `data/raw/`,
each folder with a `SOURCE.md`: `jones_molitor_reif_2019/` (wellness, CC0 1.0), `costa_ramon_schaede_slotwinski_2026/`
(teachers, CC0 1.0) and `kaufmann_andre_koszegi_2024/` (consumers, CC0 1.0 on Dataverse, CC BY-NC 4.0 in the
authors' README). The teachers source file (`3_clean_fu1.dta`) is not redistributed. The package ships instead the
base file built from it (70 numeric survey columns and the block) and the treated count;
`data_generation/build_teachers_base.R` rebuilds both, byte for byte, from the Dataverse archive after checking its
SHA-256. `wide.dta` contains four open-text answers, a free-text gender field and two respondent identifiers (the
authors replaced the Prolific IDs with random ones); the module reads none of them.

## Code that produced the agent records (`data_generation/`, not rerun)

`runtime/` is module 02's runtime with the cohort's edits (the placebo's codebooks and data, a second subscription
login, the CLI's default provider and retries), with placeholders for site paths and accounts. The first 51 originals
ran before the switch to the default provider, on a route that retried a request twice. Steps 2 to 5 ran in
`runtime/`, with the files of `data/derived/03_placebo/base/` in `base/` and module 02's `cluster/build_case_list.py`.

1. `build_teachers_base.R ZIP OUT_DIR`: the teachers extract from the authors' deposit.
2. `runtime/make_originals.py EXP_ROOT --slots 51 --seed-base 9000000 --prefix main --list main_originals.jsonl`
   (then `--first-slot 51 --list batch2_originals.jsonl`), with `runtime/generate.py`: the placebo assignment (within
   blocks at the study's treated share; for consumers, an invented module for half of the respondents), the 102
   datasets (seeds 9,000,000 + slot) and their session lists, joined as `all_originals.jsonl`.
3. `runtime/runner.py cases --list LIST --task K --tasks N --cap 1 --campaign NAME --route subscription
   --concurrency W --start-concurrency W/2 --start-gap 4`, `SUB_ACCOUNT` alternating `primary` and `second` by task:
   one session (analysis, commitment, comparison) with the prompts and the study's `codebooks/*.md` beside `data.csv`.
4. `runtime/make_references.py --arm subsample --draw raw --tail 40 --body 200 --m 0 --task N --originals-list LIST
   --n-select 51` (m = 0 means 0.10 n): banks of 2,000 subsamples, doubled up to 32,000 until both strata could
   supply their replays, and the seeded draw, listed by `build_case_list.py LIST 1000000 OUT COUNT subsample-raw`.
5. `runtime/identity_check.py OUT.json`: the cohort's search for words naming a source study (`05_identity_check.py`
   reruns it).
6. `export_cluster_records.py RUN_COPY ARCHIVE_COPY PACKAGE_ROOT`: wrote the shipped records and `bank_pvalues.csv.gz`.

Cohort: GPT-6-Luna at high reasoning effort through Codex CLI 0.157.0 in the sandbox on the University of Michigan
research cluster, signed in with two ChatGPT Pro logins; slots 0-50 ran on 30 October and 51-101 on 31 October 2026.
102 originals and 24,440 replays, all classified; slot 75 has no tail replays, because no subsample of its bank
reaches its p-value of 1.4e-11. Banks and replays ran only after a gate fixed before any original ran had passed:
at least 7 of the first 51 originals significant at 5%. A match is the replay agent's own verdict. The cohort cost no
dollars and about 44 points of the two accounts' weekly usage meters. Rerunning the sessions needs model access and
gives a new cohort.

Not included: the session archives (130 node archives, 11.2 GB; no restricted data, omitted for size). Earlier
versions, job files and helper scripts of the runs are not included and are available from the authors.
