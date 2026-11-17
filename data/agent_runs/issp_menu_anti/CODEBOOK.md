# Codebook: issp_menu_anti

Known selection rules on a fixed menu (Appendix app:immigration_all), anti-immigration analyst. GPT-6-Luna at high effort, Codex CLI 0.165.0 on a ChatGPT Pro login on the cluster, 30 October 2026; target AD061 (support for government responsibility for the unemployed on the UN immigrant-stock share, no controls), u = 0.0256. In a first array of 100 workers, 81 cases failed, 80 of them on ChatGPT rate limits; all 81 reran from scratch before any match rate was looked at, and one failed again and is excluded. No API spend. Each replay received a subsample of 13 of the 46 countries (all respondents of each), drawn without replacement within workflow and stratum from the bank: 100 per stratum for the fixed model and 200 per stratum for frontier and client pressure. All 999 recorded selections follow the workflow's rule.

## Files

- `design.json`: each workflow's target analysis (the menu model the full-data originals reported) and its full-data
  p-value u, as frozen before the replays ran.
- `originals.csv`: the six full-data originals, two per workflow (the first is the target, the second checks that the
  workflow reproduces it).
- `replays.csv`: the 1,000 replays, one row each (fields below).
- `declarations.jsonl.gz`: per session, `session_id`, `role`, the `commitment` (declared model and null) and, for
  replays, the `comparison` (the replay's own verdict and reason).
- `inputs/`: the files every session received: `ANALYSIS_CONTRACT.md`, `CATALOGUE.json` (the 108-model menu), `catalogue_helper.py` (fits any menu model), `developer-instructions.txt`, `commitment_prompt.txt`, `comparison_prompt.txt`, and `prompts/` (the workflow prompts of the anti-immigration analyst). The pro arm used the same files with its own `prompts/`.

## Fields

| Field | File | Meaning |
|---|---|---|
| `session_id` | both | Opaque case id (replays) or `<workflow>-<replicate>` (originals) |
| `role` | both | `original` or `replay` |
| `workflow` | both | `fixed_reference` (report the named model whatever its result), `frontier` (report the model with the most negative HC3 t-statistic), `client_pressure` (report the named model if it is negative and significant, otherwise the most negative alternative) |
| `replicate` | originals | 1 or 2 |
| `position` | replays | Launch position in the single random run order |
| `stratum` | replays | `tail` if the target's p-value on the subsample is at most u, else `body` |
| `bank_index` | replays | Index of the subsample in the 20,000-subset bank (`data/intermediate/07_immigration_menu/bank_anti.csv.gz`) |
| `countries` | replays | ISO codes of the 13 countries in the subsample, in the order their respondents were written |
| `model`, `effort`, `route` | both (`route`: replays only) | `gpt-6-luna`, `high`, `chatgpt_subscription:primary` (ChatGPT Pro login through the Codex CLI) |
| `status` | both | Replays: `complete`, `unavailable` (the session ended with an error other than a rate or usage limit; none in this arm), `interrupted_rate_limited` (cut by a ChatGPT rate limit, no selection); originals: `committed` |
| `rerun` | replays | 1 if the case reran from scratch after a rate-limited first attempt (before any match rate was looked at) |
| `selection_recovered_from_archive` | replays | 1 if the agent saved `selection.json` in a subfolder and the selection was read from the case archive |
| `selected_model_id` | both | Menu model the session reported (empty: no selection; such replays are excluded and counted) |
| `selected_theta`, `selected_se`, `selected_p` | both | Its slope, HC3 standard error and p-value on the session's data |
| `rule_pick` | replays | The model the workflow's rule picks on the subsample, computed programmatically |
| `rule_branch`, `branch` | replays (`branch` also in originals) | Client pressure only: the rule's branch (`aligned` if the named model has the expected sign and p < 0.05, else `pressure`) and the branch the runner sent |
| `own_match` | replays | The replay's own comparison verdict (1 = same model and null); the table scores a match by `selected_model_id` instead |
| `data_unchanged` | replays | 1 if the data file was unchanged at the end of the session |

`status` is the runner's record and does not decide inclusion: a replay is classified when `selected_model_id` is not
empty. In this arm the one replay without a selection has status `interrupted_rate_limited`.

## CATALOGUE.json

The menu the agents chose from. Top-level keys: `schema`, `status`, `source_plan_sha256`, `units`, `preprocessing`
(how respondent rows become country means), `inference` (OLS with HC3 standard errors and a t reference with
n_countries minus the design rank degrees of freedom), `unavailable_policy` (when a model counts as not estimable),
`m_reference` (an m-value over the menu, not used in the paper), and `models`, a list of the 108 menu
specifications. Each model has `model_id` (AD001 to AD108), `sequence`, `outcome` (one of the six welfare items),
`exposure` (`migstock_wb`, `migstock_un` or `migstock_oecd`), `coverage` (`measure_available` or
`all_three_observed`), `control_set` and `controls` (none, log GDP, or log GDP and the Gini index), and the declared
`working_model`, `null_hypothesis`, `population` and `sample_restrictions` the agents could commit to.
