# 04 Power design

Table 1 Panel D. The open-ended design of module 02 with a true program effect on employment at 12 months (+6.2
and +9.6 percentage points, 100 datasets each); replays use subsamples of 200 of the 2,000 participants.

## Analysis (run by `run_all.sh`, about 1.5 minutes)

| Script | Writes |
|---|---|
| `01_prespecified.py` | `data/derived/04_power/datasets.csv`: the pre-specified test's p-value and the realized Cohen's d per dataset |
| `02_replay_counts.py` | `replay_counts.csv`: u and tail and body replays and matches per usable original; checks each replay's stratum and subsample against the bank. `program_benchmark.csv`: each program's rejection rate on new data |
| `03_replay_bounds.R` | `bounds.csv`: the replay bounds per original |
| `04_panel_d.py` | `paper/tabs/simulation_panel_d.tex`, `results/numbers/04_power.json` |

Inputs: `data/raw/pathways_power/` (the 200 datasets, see its `SOURCE.md`), `data/agent_runs/power_luna_login/`
(session records, lead codes and the originals' programs, see its `CODEBOOK.md`) and `data/intermediate/04_power/`
(each program's p-values on its reference bank and on new datasets, and the effect calibration, see its
`CODEBOOK.md`). All are author-generated, CC BY 4.0.

## Code that produced the agent records (`data_generation/`, not rerun)

The files are edited copies of the runtime of the authors' earlier, unreported API power panel of 30 originals;
cluster paths, accounts and login files are replaced by placeholders. Steps 3 to 7 ran in one folder holding these
files, module 02's `cluster/make_originals.py`, `cluster/build_case_list.py` and `prompts/*`, and
`data/raw/pathways/CODEBOOK.md`.

1. `generate.py`: module 02's generator plus the treatment effect (tau = 0.28 on even slots, 0.43 on odd slots).
2. `calibrate_effect.py effect_calibration.json`: the effect sizes (no model calls; an estimated 35 minutes on
   2 cores).
3. `make_originals.py RUN RUN --n 200 --seed-base 9000000 --prefix power --list power_originals.jsonl`: the 200
   datasets of `data/raw/pathways_power/` (seeds 9,000,000 + slot) and the originals' session list.
4. `runner.py cases --list power_originals.jsonl --task K --tasks N --cap 1 --campaign NAME --route subscription
   --concurrency W --start-concurrency W/2 --start-gap 4`, with `SUB_ACCOUNT=second`: one agent session in a sandbox
   (analysis, commitment) and the score of its program.
5. `code_headlines.py power_originals.jsonl headline_codes.json`: the lead codes in `originals.csv`, from the
   programs and commitments, before any bank.
6. `bank_chunks.py`, with `fast_guest.py` and `make_references.py`: `plan --out PLAN.jsonl`, then
   `score --plan PLAN.jsonl --task K` per chunk and `settle --task N` and `build --task N` per original. These give
   each program's p-values on its reference subsamples and on 4,000 new datasets with and without its effect, the
   bank (2,000 subsamples, doubled until 40 tail and 200 body, up to 128,000) and the draw of the replayed subsamples.
7. `build_case_list.py power_originals.jsonl 200 power_cases.jsonl 46808 subsample-raw`, then `runner.py` as in
   step 4 with `SUB_ACCOUNT` set to `second` or `primary` by array task: the replays.
8. `extract_records.py RUN_COPY`: wrote the shipped records, bank and draw files from a copy of the run folder.

Cohort: GPT-6-Luna at high reasoning effort through Codex CLI 0.157.0, in a bubblewrap sandbox on a Linux cluster,
4 and 5 November 2026, on two ChatGPT Pro logins (originals on one, replays on both), with module 02's prompts and
codebook. 198 of the 200 originals are usable (slots 36 and 47: the agent's own program fails on its data). All
46,808 replays were classified, 40 tail and 200 body per original, except at +9.6 points, where 31 banks held fewer
than 40 tail subsamples and 3 (slots 29, 71 and 121) none, which leaves their bounds vacuous; attempts cut by rate
limits or unfinished turns were rerun on the same subsample. A match is the replay agent's own verdict. A lead's
null is true when its outcome is one the effect does not reach (`lead_family`) or its commitment states an
interaction null. The cohort cost no dollars; rerunning the replays takes about 51% of one login's weekly meter and
gives a new cohort, since agents sample their outputs.

Not included: the session archives (65 tar files, about 19.5 GiB, kept by the authors). Earlier versions, job files
and helper scripts of the runs are not included and are available from the authors.
