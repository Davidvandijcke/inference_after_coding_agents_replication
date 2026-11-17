# 05 Persona design of Miao et al. (2026)

Tables 2 and 4. Analyst agents rerun the persona design of Miao, Pritchard and Zou (2026) on the ISSP immigration
question: each run gets an anti- or pro-immigration persona as its whole system prompt and their task prompt, and its
final result goes through their inference repair and three reviews. The module also builds the restricted ISSP file.

## Analysis

`run.sh` takes about 2 seconds.

| Script | Writes |
|---|---|
| `01_read_sfig3c.py` | `data/derived/05_immigration_personas/miao_sfig3c_z.csv`: the z of each run in Miao et al.'s Supplementary Figure 3c |
| `02_persona_tables.py` | `paper/tabs/miao_personas.tex`, `paper/tabs/miao_personas_original.tex`, `results/numbers/05_immigration_personas.json` |

Inputs: the run records in `data/agent_runs/persona_*/` and the page of Miao et al.'s Supplementary Figure 3 in
`data/raw/miao2026/paper/`. A run counts when at least two of its three reviews pass it; its z comes from the
repaired p-value.

## Code that produced the agent records (`data_generation/`, not rerun)

1. `build_issp_file.py`: the restricted ISSP analysis file of modules 05 to 07 (command in `data/raw/issp/SOURCE.md`).
2. `analysts/run_miao.mjs` (Claude) and `analysts/run_luna_codex.py` (GPT-6-Luna): one analyst run per persona, with
   the persona and task prompts of their repository (`data/raw/miao2026/SOURCE.md`) and our edit of its data prompt,
   `analysts/prompts/issp_paper.md`. GPT-6-Luna on Miao et al.'s file ran a version of `run_luna_codex.py` adapted to
   module 02's `cluster/runner.py`.
3. `repair_review/r1_repair.mjs`: their inference repair of each final result, with our changes marked `ALIGN:`.
4. `repair_review/r2_sol.py` with `r2_prompts.py` and `prompts/` (their Appendix B): the summary and three reviews.
5. `extract_persona_records.py`: wrote the six cohort folders from the run folders.

The analysts are Claude Sonnet 4.6 and 5.5 at the SDK's default effort and GPT-6-Luna at high effort; each cohort is
described in `data/agent_runs/persona_*/CODEBOOK.md`. Rerunning the runs needs model access.
