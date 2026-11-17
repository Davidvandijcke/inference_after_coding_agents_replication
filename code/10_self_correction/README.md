# 10 Agents' own corrections

Table 9, Figures 9 and 10 and the in-text numbers on agents' own multiplicity corrections. Four agents analyze the
open-ended design's null datasets, GPT-6.1-Sol analyzes placebo versions of the Card-Krueger data, and Sol's own
correction is compared with the conditional correction on datasets with a planted effect.

## Analysis

`run.sh` takes about 15 seconds.

| Script | Writes |
|---|---|
| `01_stance_table.py` | `paper/tabs/self_correction_stance.tex`: significant leads, stance, tests stated and corrections per agent |
| `02_card_sol_interval.R` | `data/derived/10_self_correction/card_sol_interval.csv`: replay correction of Sol's Card original |
| `03_card_family.py` | `paper/figs/self_correction_family.pdf`, `card_family_coding.csv`: replay declarations coded against the original's 26 analyses |
| `04_power_verdicts.py` | `paper/figs/self_correction_verdicts.pdf`, `power_cases.csv`: own and conditional verdicts per planted dataset |
| `05_numbers.py` | `results/numbers/10_self_correction.json`, from the `numbers_*.json` of scripts 01, 03 and 04 |

Inputs: the records in `data/agent_runs/multi_model_originals/`, `card_placebo_sol/` and `sol_common_analysis/`,
module 02's `open_ended_luna/` and module 09's `card_sol/`. A Card replay's match is module 09's review label; a null
reference run counts in Figure 10 when it ran the common analysis and a blind reader matched its declaration.

## Code that produced the agent records (`data_generation/`, not rerun)

1. Datasets: module 02's null datasets (Table 9), module 04's `generate.py` with a planted effect or none
   (Figure 10), and `card_placebo/frozen/generate.py`, which permutes the Card data's state labels (Figure 9).
2. `four_agent_null_run/run_codex.py` (GPT-6.1-Sol) and `run_claude.py` (Claude, agent shell `clean_bash.sh`), with
   `common.py` and module 02's `prompts/`: one sandboxed session per dataset (analysis, then commitment).
3. Figure 10's runs used the Sol runner with `sol_common_analysis/frozen/research_prompt.txt`; Figure 9's used it with
   `card_placebo/frozen/research_prompt.txt`, `CODEBOOK.md` and a version of `common.py` adapted to the Card data.
4. Blind reads: `four_agent_null_run/review.py` under `review2_rule_stance.txt` and `review2_rule_facts.txt`; a
   version of it under `card_placebo/reader_prompt.txt` and `reader_schema.json` for the Card placebo reports.
5. `sol_common_analysis/reference_runs/READER_PROMPT_FROZEN.txt` and `READER_SCHEMA_USED.json`: a version of
   `review.py` reading 20 items per session checked the reference runs' declarations against the common analysis.
6. `extract_records.py`, with `four_agent_null_run/code_choices.py` coding each program's choices: wrote the shipped
   records and regenerated the datasets.

The agents are GPT-6.1-Sol, Claude Sonnet 5.5 and Claude Opus 5.5 at high reasoning effort, and GPT-6.1-Sol at high
effort did every blind read; each cohort is described in the `CODEBOOK.md` of its records folder. Rerunning the
sessions needs model access and gives new records, since agents sample their outputs.
