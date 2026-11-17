# Codebook: persona_sonnet46_corrected

Analyst: Claude Sonnet 4.6 at the SDK's default effort, Claude Agent SDK 0.3.197. Data: the corrected ISSP file (`issp_clean.csv`). Runs: 20 anti- and 20 pro-immigration, all completed and reviewed. Dates and route: runs 1-10 per persona on 27 October 2026 (OpenRouter API route), runs 11-20 on 29 October 2026 (Claude subscription, on the cluster). Runs 11-15 per persona were redone from scratch on the cluster with the same driver, prompts, data and versions after a first attempt on a Mac was stopped; the records hold the redone runs. The repair job of this cohort passed `data_clean.csv` as the data path: 39 of the 40 repair agents found `issp_clean.csv` themselves, and one (pro-immigration run 05) derived its t(G-1) p-value from the saved standard error without running the script.

## Design

Miao, Pritchard and Zou (2026), persona design for the ISSP immigration question: each analyst run receives a persona
(anti- or pro-immigration) as its whole system prompt and the authors' task prompt (ten rounds of ten models over
seven axes, then a final result); neutral-persona runs are not included. The data prompt in it
(`code/05_immigration_personas/data_generation/analysts/prompts/issp_paper.md`) is their `prompts/data/issp.md` with
the sign convention of their paper, in which a positive AME
means more support, in place of the repository's opposite convention and its instruction to flip signs, and without
the identifier `provisional`. Every final result then went through Miao et al.'s inference repair (their
`Round1_reviewer.mjs`, run with Claude Sonnet 4.6 for every cohort), a structured summary of the repaired
specification and three independent PASS/FAIL reviews of that summary (GPT-6.1-Sol, high effort, Codex CLI 0.165.0
on a ChatGPT Pro login), all on 4-5 November 2026. The review prompt (with the corrected file's 154,921 respondents
in place of their 152,594) and the two example summaries are their Appendix B's; the rest of the summary prompt and
the reviewers' output-format instruction are ours (`repair_review/r2_prompts.py`), as theirs are not published.
Programs: `code/05_immigration_personas/data_generation/`.

## Files

- `records.csv`: one row per run, fields below.
- `texts.jsonl.gz`: one JSON object per run with `session_id` and the texts `analyst_program` (the analyst's final
  script), `repair_report` (the repair agent's review), `repaired_program`, `summary` (the specification summary the
  reviewers saw) and `review_1` to `review_3` (the reviewers' full answers). Absolute paths that the agents
  hard-coded in their programs and reports are replaced by placeholders (`<cluster scratch>`, `<local runtime>`,
  `<local folder>`, `<temporary folder>`).

## Fields of records.csv

| Field | Meaning |
|---|---|
| `session_id` | Run identifier: cohort/persona/run folder, or the session's case id for GPT-6-Luna on Miao et al.'s file |
| `persona` | `anti` or `pro` (immigration) |
| `analyst_model` | Model of the analyst run |
| `analyst_effort` | Reasoning effort: from the run record for GPT-6-Luna; `SDK default` for the Claude Agent SDK, whose default is high (the driver sets none) |
| `analyst_harness` | Agent harness and version |
| `analyst_route` | `subscription` (Claude or ChatGPT login) or `openrouter` (paid API) |
| `analyst_date` | Date the analyst run started (UTC): from the run's metadata, or for GPT-6-Luna on the corrected file from its Codex thread id (a UUIDv7) |
| `data_file` | `corrected` (`issp_clean.csv`, SHA-256 bd2363ce...) or `miao` (`data_clean.csv` from Miao et al.'s repository, SHA-256 90e1b627...) |
| `status` | `completed`: every run here produced a final result, a repair and three reviews |
| `analyst_dep_var`, `analyst_immigration_measure`, `analyst_model_type`, `analyst_controls`, `analyst_n_obs` | The analyst's final specification as it reported it (`results_final.csv`, row `final`) |
| `analyst_ame`, `analyst_ame_se`, `analyst_ame_pvalue` | The analyst's reported average marginal effect of immigration on welfare support, its standard error and two-sided p-value (positive = more support) |
| `repaired_*` | The same fields after the inference repair (`results_final_revised.csv`); the tables use `repaired_ame` and `repaired_ame_pvalue` |
| `repair_model` | Model of the inference repair: `claude-sonnet-4-6` for every cohort (from the run plan and the repair script's default; no per-run record) |
| `review_model` | Model of the specification summary and the three reviews: `gpt-6.1-sol`, high effort |
| `review_1`, `review_2`, `review_3` | PASS or FAIL, from the first line of each review |
