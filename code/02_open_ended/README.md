# 02 Open-ended agent design

Table 1 Panel B and Figure 3. A coding agent analyzes a synthetic randomized evaluation in which every treatment
effect is zero and reports a p-value; replay agents repeat the task on subsamples of 200 of its 2,000 participants.

## Analysis (run by `run_all.sh`, about 2 minutes)

| Script | Writes |
|---|---|
| `01_replay_counts.py` | `data/derived/02_open_ended/replay_counts.csv`: replays and matches per original; checks each session's data and stratum |
| `02_replay_intervals.R` | `per_original.csv`: corrected p-value, multiplier and intervals per original |
| `03_panel_b_figure.py` | `paper/tabs/simulation_panel_b.tex`, `paper/figs/simulation_overview.pdf`, `results/numbers/02_open_ended.json` |

Inputs: `data/raw/pathways/` (the 100 datasets and their codebook), `data/agent_runs/open_ended_luna/` (session
records, see its `CODEBOOK.md`), `data/intermediate/02_open_ended/bank_pvalues.csv.gz` (each original program's
p-value on the subsamples of its reference bank), and the derived outputs of modules 01 and 03 for Figure 3(a).

## Code that produced the agent records (`data_generation/`, not rerun)

Steps 2 to 7 ran in one folder holding the `cluster/` scripts, `generate.py`, `prompts/*` and
`data/raw/pathways/CODEBOOK.md`; `runner.py` reads the site paths (run folder, CLI build, proxy) from the environment.

1. `generate.py`: the data-generating process, in which every treatment effect is zero.
2. `cluster/make_originals.py OUT EXP --n 100 --seed-base 8000000 --prefix main --list main_originals.jsonl`: the
   100 datasets of `data/raw/pathways/` (seeds 8,000,000 + slot) and the originals' session list.
3. `prompts/`: the research, commitment and comparison prompts.
4. `cluster/runner.py cases --list LIST --task N --tasks K --cap CAP --campaign NAME --concurrency 32`: runs each
   session in a sandbox (analysis, commitment and, for a replay, comparison) and scores each original's program. The
   second batch ran copies set to the flex tier with a longer stream timeout, on the OpenAI key and then through
   OpenRouter (other price and key files, `route` label `openrouter-flex`); in its last three waves the copy also
   reran refused or cut-off turns. The copies used before those three waves were not saved.
5. `cluster/make_references.py --arm subsample --draw raw --m 200 --tail 40 --body 200 --task N --originals-list LIST
   --n-select K`, one task per usable original: the reference bank (2,000 subsamples scored with the original's
   program, doubled until both strata can supply their draws) and the seeded draw of 40 tail and 200 body replays.
6. `cluster/build_case_list.py LIST K OUT EXPECTED subsample-raw`: the shuffled list of drawn replays (23,729 in
   all), run with `runner.py cases`.
7. `cluster/extend_references.py --task N --originals-list LIST --n-select 99`: extends each bank and draws 200 more
   tail and 200 more body replays among subsamples not drawn before; `build_case_list.py` with the arm
   `subsample-raw-ext1` lists the 39,434 drawn, run as in step 4.
8. `extract_records.py RESEARCH NODE_RESULTS`: wrote the shipped records, programs and `bank_pvalues.csv.gz`.

Cohort: GPT-6-Luna at high reasoning effort through Codex CLI 0.157.0, in a bubblewrap sandbox on a Linux cluster,
27 to 30 October 2026, on the OpenAI API (the second batch of replays at the flex tier and through OpenRouter).
100 originals ran and 99 are usable; 23,729 replays were drawn first and 39,434 more after the first results were
seen. A match is the replay agent's own verdict. Rerunning the sessions needs model access, costs about $1,000 and
gives a new cohort, since agents sample their outputs.

Not included: the billing relay that `runner.py` starts on the API route (accounts and provider traffic only) and
the full session archives (26 GB, kept by the authors). Earlier versions, job files and helper scripts of the runs
are not included and are available from the authors.
