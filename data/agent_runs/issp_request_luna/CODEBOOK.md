# Codebook: issp_request_luna

The realistic request of the immigration application (main text, "Realistic selection pressure"; Figure 4,
`fig:issp_results`). GPT-6-Luna at high effort, Codex CLI 0.157 in a network-restricted sandbox on the cluster,
API route through a cost-capped relay, 27-28 October 2026. Every session received the research prompt, the codebook
and a data file (`inputs/`): the originals the full corrected ISSP file (154,921 respondents, 46 countries) in a
seeded row order, the replays a subsample of 16 whole countries. After its analysis each session received the
commitment prompt; each replay then received the comparison prompt with the original's commitment.

The first replay jobs completed 536 replays before the OpenAI account ran out of credit; a rerun job ran the other
464 from scratch on the same subsamples (`completed_in_rerun_job`; `retried_after_usage_limit` marks the 201 that a
usage limit refused before the agent acted). Spend recorded by the relay: $52.85. All 1,000 replays completed and were
classified; no session was excluded.

All files are written by `code/06_immigration_request/data_generation/extract_request_records.py` from the run
records in the research repository (`output/issp-luna-selection-20261027/`): `inputs/` is copied from its `source/`,
`review_candidates.csv` from `review/wording_candidates.jsonl`, and `original/programs/analysis_final.py` from the
original's workspace copy (`output/issp-application-20261028/original/emitted-analysis_final.py`).

## Files

- `inputs/`: `research_prompt.txt`, `CODEBOOK.md` (the agent-facing codebook), `commitment_prompt.txt`,
  `comparison_prompt.txt`.
- `originals.csv`: the three full-data originals; slot 0 is the original the paper corrects.
- `replays.csv`: the 1,000 replays, one row each (fields below).
- `declarations.jsonl.gz`: per session, `session_id`, `role`, the `commitment` (declared `model`, `null`,
  `outcome`, `population`, `sample_restrictions`) and, for replays, the `comparison` (`model_null_match`, `reason`).
- `review_candidates.csv`: the 495 nonmatching replays that the second reading's keyword rules picked as
  candidates, of which 86 were read (below) (`i` is the index used by
  `code/06_immigration_request/data_generation/review/build_reviewed.py`; `session_id`; `stratum`).
- `original/`: the selected original's records, described below.

## The original's records

- `ANALYSIS_PATHS.jsonl`: the agent's append-only analysis log, one JSON event per line. Every event has
  `event_id`, `event_type` and `path_id` (the analysis it belongs to). Event types and their other keys:
  - `initial_plan`: `data_inventory`, `descriptive_scope_notes`.
  - `plan`: `question`, `null`, `population_sample_rules`, `outcome_exposure_construction`, `controls`, `model`
    (the regression formula), `weighting`, `uncertainty_method`, `saved_code_path`, `parent_path_id`, `reason`.
  - `result`: `status` (`estimated` or `failed`), `estimate_raw`, `std_error_raw`, `ci95_raw`,
    `estimate_per_1pp_stock`, `std_error_per_1pp_stock`, `ci95_per_1pp_stock`, the same on the 0-100 outcome scale
    and per 10 points of stock (`*_on_0_100_outcome_scale`, `*_per_10pp_stock_on_0_100_outcome_scale`), `p_value`,
    `sample`, `diagnostics`, `exposure_unit_comparison`, `reason`.
  - `next_choice`: `choice` (the next analysis), `prior_path_ids`, `reason` (the agent's stated reason).
  - `correction`: `original_event_id`, `correction` (four failed results corrected and rerun).
  - `final_choice`: `alternative_path_ids_considered`, `rationale`.

  The log names the cluster's Python interpreter in four failed commands; that path is replaced by
  `<cluster python>`.
- `results_final.json`: the reported result: `path_id`, `question`, `null`, `model`, `estimate` and `std_error`
  (points on the 0-100 scale per 10 points of immigrant stock), `ci95`, `p_value`, `estimate_unit`,
  `positive_direction`, `raw_coefficient_per_1pp_stock`, `raw_std_error_per_1pp_stock`, `sample` (respondents,
  countries, country-waves, years, complete-case variables, weighting) and `uncertainty`.
- `BRIEF.md` (the one-page brief), `final_report.md` (the closing message) and `programs/analysis_final.py` (the
  frozen program, which scores every bank subsample).

## Fields of originals.csv and replays.csv

| Field | File | Meaning |
|---|---|---|
| `session_id` | both | Opaque case id |
| `role` | both | `original` or `replay` |
| `slot` | originals | Launch slot of the candidate original |
| `model`, `effort`, `route` | both | `gpt-6-luna`, `high`, `api` |
| `status` | both | `completed` when the analysis, commitment and (replays) comparison phases all finished |
| `data_sha256` | both | SHA-256 of the data file the session received; for replays it equals the bank subsample's (`data/intermediate/06_immigration_request/bank_p.csv`) |
| `program_p` | originals | p-value of the session's own program on its own data; slot 0's is the original's u |
| `selected_as_original` | originals | 1 for the original the replays target |
| `stratum` | replays | `tail` if the original program's p-value on the subsample is at most u, else `body` |
| `bank_index` | replays | Index of the subsample in the 4,000-subsample bank (countries drawn by the seed rule in `code/06_immigration_request/data_generation/score_bank.py`) |
| `subsample_p` | replays | The original program's p-value on the subsample |
| `retried_after_usage_limit` | replays | 1 if a first attempt was refused for a usage limit before the agent acted |
| `completed_in_rerun_job` | replays | 1 for the 464 replays completed by the rerun job (the 201 above and 263 not completed when the first jobs stalled) |
| `own_match` | replays | The replay's own comparison verdict (1 = same model and null as the original) |
| `reviewed_match` | replays | Headline verdict: `own_match`, or 1 where the second reading found that the only stated differences were wording (`review_category`) |
| `review_category` | replays | For the 77 nonmatches the second reading listed: `dependence` (the replay declared within-country dependence, which the original left to its inference procedure), `linearity` ("linear conditional mean" against "linear projection"), `redundant` (a complete-case rule implied by the others, or a rescaled contrast); combinations joined by `+` |
| `review_uncertain` | replays | 1 for the four replays (all tail) whose population wording could restrict the target to the realized countries; the reading left them uncertain and they count as nonmatches |
| `review_reader` | replays | Reader of the second reading: a Claude Code session running Claude Opus 5.5 |

The second reading was done on 28 October 2026 by a Claude Code session. Keyword rules over the comparison reasons
of the 962 nonmatches picked 495 candidates (`review_candidates.csv`) and narrowed them to 86. The session read
those 86 reasons and 4 from outside the candidates, each printed with its stratum, so the reading was not blind to
stratum; it read the reason each replay gave for its own verdict (`comparison.reason` in `declarations.jsonl.gz`),
not the two declarations. It counted 73 nonmatches as matches, 72 of the 86 and one from outside (case `dd475b...`,
a redundant complete-case rule whose reason mentions neither dependence nor linearity), and left 4 uncertain. By
`review_category`, the 73 are `dependence` 40, `linearity` 13, `dependence+linearity` 17 and `redundant` 3. The other
409 candidates and the 467 nonmatches outside the candidate list were not read, so the reading is partial:
nonmatches with stated differences of the same kinds may remain among them, labeled as nonmatches. No code for the
keyword rules exists.
