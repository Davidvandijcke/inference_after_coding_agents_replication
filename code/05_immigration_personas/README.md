# 05 Persona design of Miao et al. (2026)

Tables 2 and 4. Analyst agents rerun the persona design of Miao, Pritchard and Zou (2026) on the ISSP immigration
question: each run receives an anti- or pro-immigration persona as its whole system prompt and their task prompt (ten
rounds of ten models over seven axes, then a final result), and each final result goes through their inference repair
and three reviews. The module also holds the build of the restricted ISSP analysis file that modules 05 to 07 use.

## Analysis (run by `run_all.sh`, about 2 seconds)

| Script | Writes |
|---|---|
| `01_read_sfig3c.py` | `data/derived/05_immigration_personas/miao_sfig3c_z.csv`: the z-statistic of each run plotted in Miao et al.'s Supplementary Figure 3c, read from the page's vector graphics |
| `02_persona_tables.py` | `paper/tabs/miao_personas.tex`, `paper/tabs/miao_personas_original.tex`, `results/numbers/05_immigration_personas.json` |

Inputs: `data/agent_runs/persona_*/records.csv` (one row per anti- or pro-immigration run; see each cohort's
`CODEBOOK.md`) and `data/raw/miao2026/paper/supplementary_figure_3_page35.svg` (see that folder's `SOURCE.md`).

## Code that produced the agent records (`data_generation/`, not rerun)

1. `build_issp_file.py`: builds the restricted ISSP analysis file (command in `data/raw/issp/SOURCE.md`).
2. `analysts/requirements.txt` (the analysts' Python, in `$MIAO_RUNTIME/venv/`) and `analysts/prompts/` (our data
   prompts, with the paper's sign convention); add Miao et al.'s persona and round files from `data/raw/miao2026/`.
3. `analysts/run_miao.mjs` (Claude analysts; default data: Miao et al.'s file): `node run_miao.mjs --model
   claude-sonnet-4-6 --runs 20 --personas anti_immigration,pro_immigration --data issp_clean.csv --data_prompt
   issp_paper_clean.md`; `analysts/run_luna_codex.py` (GPT-6-Luna, corrected file) takes `--personas` and `--runs`.
4. `analysts/luna_miao_file/miao_pilot.py` with `runner.py` and `prompts/` (add `issp_paper.md`, `_round_template.md`,
   `persona/`), GPT-6-Luna on Miao et al.'s file, `EXP_ROOT` holding `data/data_clean.csv`: `make-cases --runs 4
   --delivery system --out cases.jsonl`, then `run --list cases.jsonl --results results_system.jsonl`.
5. `repair_review/stage_runs.py`: copies each run's final program and result; steps 6 and 7 read them from
   `base/<cohort>/results/`, beside the cohort's data file in `base/<cohort>/data/`.
6. `repair_review/r1_repair.mjs`, their repair with our changes marked `ALIGN:` (their `strip_py.py` beside it):
   `MIAO_RUNTIME=<runtime> BASE_DIR=base/<cohort> DATA_PATH=<csv> ENV_FILE=<token> node r1_repair.mjs --role R --run N`.
7. `repair_review/r2_sol.py` with `r2_prompts.py` and `prompts/` (the paper's Appendix B): `python r2_sol.py --cohort
   <cohort> --phase summary --account second`, then `--phase review`.
8. `extract_persona_records.py <research repository>`: wrote the six cohort folders from the run folders.

Cohorts, one folder each in `data/agent_runs/` (neutral-persona runs are not included):

| Folder | Analyst | Harness | Data | Runs (anti/pro) | Dates | Route |
|---|---|---|---|---|---|---|
| `persona_sonnet46_corrected` | Claude Sonnet 4.6 | Claude Agent SDK 0.3.197 | corrected file | 20/20 | 27 and 29 Oct 2026 | OpenRouter API (runs 1-10), Claude subscription (11-20) |
| `persona_sonnet55_corrected` | Claude Sonnet 5.5 | Claude Agent SDK 0.3.197 | corrected file | 20/20 | 29 Oct 2026 | Claude subscription |
| `persona_luna_corrected` | GPT-6-Luna, high effort | Codex CLI 0.164.0-alpha.11.1 | corrected file | 20/20 | 29 Oct 2026 | ChatGPT subscription |
| `persona_sonnet46_miaofile` | Claude Sonnet 4.6 | Claude Agent SDK 0.3.197 | Miao et al.'s file | 20/20 | 27 Oct 2026 | Claude subscription, OpenRouter API for 5 runs |
| `persona_sonnet55_miaofile` | Claude Sonnet 5.5 | Claude Agent SDK 0.3.197 | Miao et al.'s file | 10/10 | 27 Oct 2026 | Claude subscription |
| `persona_luna_miaofile` | GPT-6-Luna, high effort | Codex CLI 0.157 | Miao et al.'s file | 4/4 | 27 Oct 2026 | ChatGPT subscription |

The Claude analysts ran at the SDK's default effort. Every run completed; runs that a first attempt on a Mac left
unfinished were redone from scratch with the same driver, prompts, data and versions. On 4 and 5 November 2026 each
final result went through Miao et al.'s inference repair with Claude Sonnet 4.6 for every cohort (Claude
subscription), then a summary of the repaired specification and three independent PASS/FAIL reviews of it by
GPT-6.1-Sol (high effort, Codex CLI 0.165.0, ChatGPT Pro login), with their Appendix B review prompt and a summary
prompt built from that appendix's two example summaries (theirs is not published). The repair job for Claude Sonnet
4.6 on the corrected file passed `data_clean.csv` as the data path: 39 of the 40 repair agents found `issp_clean.csv`
themselves, and one (pro-immigration run 05) derived its t(G-1) p-value from the saved standard error without running
the script. A run is kept when at least two reviews pass it. Its z comes from the repaired p-value; the one kept run
whose repaired p-value is written as 0 (Claude Sonnet 4.6 on Miao et al.'s file, anti-immigration run 04) counts as
significant and has no z. `records.csv` takes the repair and review models from the run plan and script settings.

Not included: the ISSP microdata and Miao et al.'s ISSP file (see `data/raw/issp/SOURCE.md`) and the analysts'
message streams, whose command output may print respondent rows. Paths and accounts are placeholders. Earlier
versions, job files and helper scripts of the runs are not included and are available from the authors.
