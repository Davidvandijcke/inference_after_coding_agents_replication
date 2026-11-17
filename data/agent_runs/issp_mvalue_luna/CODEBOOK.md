# Codebook: issp_mvalue_luna

Reference sessions for Miao et al.'s m-value of the realistic request's original (Appendix app:immigration,
"Reference distribution for the m-value"). 60 full-data GPT-6-Luna sessions at high effort, 29 October 2026, with
the original's runner, Codex CLI 0.157 build, sandbox, codebook and data (the full corrected ISSP file in a seeded row
order per session); analysis phase only. 20 sessions per brief: `pro_team` (the original's research prompt,
`../issp_request_luna/inputs/research_prompt.txt`), `anti_team` (`inputs/research_prompt_anti.txt`) and
`nonpartisan` (`inputs/research_prompt_nonpartisan.txt`); the variants change one sentence. API route through the
cost-capped relay; spend $2.04 by the relay's estimate. All 60 completed on the first attempt.

## Files

- `sessions.csv`: one row per session.
- `analysis_paths.jsonl.gz`: one JSON object per session: `session_id`, `analysis_paths` (the session's
  append-only analysis log, `ANALYSIS_PATHS.jsonl`, as text: plans, results with estimates, standard errors and
  p-values, corrections, next-step choices and the final choice) and `results_final` (the session's final result
  file, as text). `code/06_immigration_request/03_mvalue.py` reads every estimated analysis from these logs.

## Fields of sessions.csv

| Field | Meaning |
|---|---|
| `session_id` | Opaque case id |
| `role` | `reference` |
| `brief` | `pro_team`, `anti_team` or `nonpartisan` |
| `session` | Session number within the brief (1-20) |
| `prompt_file` | Research prompt the session received |
| `model`, `effort`, `route` | `gpt-6-luna`, `high`, `api` |
| `status` | `completed` when the analysis phase finished |
| `data_sha256` | SHA-256 of the data file the session received |
