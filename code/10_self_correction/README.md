# 10 Agents' own corrections

Table 9, Figures 9 and 10 and the in-text numbers on agents' own multiplicity corrections: four agents' reports on
the open-ended design's 100 null datasets, the tests GPT-6.1-Sol counts on Card-Krueger data, and its own correction
against the conditional correction on datasets with a planted effect.

## Analysis (run by `run_all.sh`, about 15 seconds)

| Script | Writes |
|---|---|
| `01_stance_table.py` | `paper/tabs/self_correction_stance.tex`: per agent, significant leads, stance of the opening, tests stated, corrections |
| `02_card_sol_interval.R` | `data/derived/10_self_correction/card_sol_interval.csv`: replay correction of Sol's Card original |
| `03_card_family.py` | `paper/figs/self_correction_family.pdf`, `card_family_coding.csv`: keyword coding of each replay declaration against the original's 26 analyses |
| `04_power_verdicts.py` | `paper/figs/self_correction_verdicts.pdf`, `power_cases.csv`: own and conditional verdicts per planted dataset |
| `05_numbers.py` | `results/numbers/10_self_correction.json`, from the `numbers_*.json` that scripts 01, 03 and 04 write |

Inputs: `data/agent_runs/multi_model_originals/`, `card_placebo_sol/` and `sol_common_analysis/` (see their
`CODEBOOK.md`); `open_ended_luna/originals.csv` and `reports.jsonl.gz` (module 02); `card_sol/original.json`,
`original/all_analyses.csv`, `replays.csv.gz` and `report_reads.csv` (module 09). The datasets the sessions analyzed
(`data/raw/pathways/`, `pathways_sol_common/`, `card_placebo/`, see their `SOURCE.md`) are not read by `run.sh`.

## Code that produced the agent records (`data_generation/`, not rerun)

1. `four_agent_null_run/build_workspaces.py`: rebuilds module 02's 100 null datasets (with its `generate.py` placed
   in `frozen/`) and writes the roster of Table 9's sessions.
2. `four_agent_null_run/run_codex.py --model gpt-6.1-sol --account primary` and `run_claude.py --model M` (agent shell
   `clean_bash.sh`), each with `--campaign C --task T --tasks 2`, `common.py` and the inputs in `frozen/`.
3. `four_agent_null_run/build_review2.py` (with `build_review.py`'s opening span and `review2_rule_*.txt`, run under
   `analysis.sbatch`), then `review.py` through `review.sbatch` with `MMO_REVIEW=review2`: the blind reads.
4. `sol_common_analysis/make_twin.py panel|panel2 gpt-6.1-sol OUT CODE POWER` (`frozen/` as `CODE/frozen_noexplore/`):
   Figure 10's planted datasets and rosters; sessions as in step 2; reads under `review2_rule_facts.txt` by `review.py`.
5. `sol_common_analysis/reference_runs/run_codex.py --model gpt-6.1-sol --account primary` with `common.py` and
   `MMO_MAX_ATTEMPTS=1`: reference sessions on step 4's `frozen/`, datasets by `frozen_reference/generate_case.py`.
6. `reference_runs/collect_and_score.py ROOT` (with `four_agent_null_run/code_choices.py` beside it) marks the common
   procedure; the blind reads come from `build_reader_batches.py`, `reader.py ROOT` (rules `READER_*`) and
   `integrate_reader_labels.py`.
7. `card_placebo/run_codex.py --model gpt-6.1-sol --account A --campaign C` with `common.py`, `frozen/` (plus the
   Card data file as `source_card.csv`) and `CCS_CONTROL`: Figure 9's placebo sessions, 40 pilot datasets, then 160.
8. `card_placebo/build_blind_review.py --record-root RUN --out WAVE` (`reader_prompt.txt`), then `run_blind_review.py
   --account A` with `CCS_ROOT=RUN`, `CCS_REVIEW=WAVE`, `MMO_PHASE_TIMEOUT=300` and `RUN/scoring/reader_schema.json`.
9. `extract_records.py RESEARCH/output STAGING`: the record and dataset folders, from runner records and raw reads
   joined to their keys, with `four_agent_null_run/code_choices.py`; it checks each regenerated dataset's hash.

The rosters of the reference and placebo runs and the builder of the planted reads' packets are not in the package.
Earlier versions, job files and helper scripts of the runs are not included and are available from the authors.

All sessions ran at high reasoning effort on subscription logins, with no paid API (the Sonnet and Opus originals cost
about $70 at list prices); rerunning one gives new records, since agents sample their outputs. All reads are by
GPT-6.1-Sol at high effort under rules frozen before the reads, and blind: the reader saw no agent name, dataset or
program output. Cluster paths and accounts in the code are placeholders. Synthetic data and agent records are CC BY
4.0; the Card placebo datasets keep the terms of `data/raw/card_krueger/`.

- `multi_model_originals`: GPT-6.1-Sol (Codex CLI 0.165, ChatGPT Pro), Claude Sonnet 5.5 and Claude Opus 5.5 (Claude
  Code CLI 2.1.314, Claude Max), 30 October 2026, 100 sessions each (usable 100, 100, 97); Table 9's GPT-6-Luna row is
  module 02's cohort on the OpenAI API. 1,320 reads label each opening's stance (whether it gives a concrete reason
  to doubt the lead) and the tests and correction each report states. The stance rule was revised after a first
  blind read of the same openings, which had counted words the prompt itself asked for as hedges.
- `card_placebo_sol`: GPT-6.1-Sol (Codex CLI 0.165, two ChatGPT Pro logins), 2 November 2026, 200 sessions; a read
  gives the tests each report states (the 40 pilot reports were read again under a corrected prompt).
- `sol_common_analysis`: GPT-6.1-Sol (Codex CLI 0.165, ChatGPT Pro logins), no-explore prompt. 60 planted sessions,
  1 November 2026 (the second 30 added after the first 30 had been read), 49 of which report the common analysis; a
  read gives each report's own correction. 100 null reference sessions, 2 November 2026: 59 run the common
  procedure, and a reader matched 51 of their declarations to it.
