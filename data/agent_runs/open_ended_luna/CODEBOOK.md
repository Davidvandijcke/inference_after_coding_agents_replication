# Open-ended agent design, GPT-6-Luna cohort

Records of the open-ended agent design (Table 1 Panel B, Figure 3): one original session per dataset in
`data/raw/pathways/` (100 sessions, 99 usable), and replay sessions on subsamples of 200 of each usable original's
2,000 participants: 40 tail and 200 body replays per original (`base`), then 200 and 200 more (`ext1`), drawn after
the first results were seen. The agent is GPT-6-Luna at high reasoning effort through Codex CLI 0.157.0 in a
bubblewrap sandbox on a Linux cluster, 27 to 30 October 2026, on the OpenAI API (the second batch at the flex tier
and through OpenRouter). Prompts: `code/02_open_ended/data_generation/prompts/`. Author-generated, CC BY 4.0.

## `originals.csv` (100 rows)

| Field | Meaning |
|---|---|
| `slot` | Original index 0-99; dataset `data/raw/pathways/main-SS.csv` |
| `seed` | Generator seed of the dataset, 8,000,000 + slot |
| `dataset` | File name in `data/raw/pathways/` |
| `original_id` | Opaque session id |
| `model`, `effort`, `harness`, `route` | `gpt-6-luna`, `high`, `codex-cli 0.157.0`, `api` |
| `status` | `completed` for all 100 |
| `usable` | `FALSE` for slot 27, whose program fails on its own data |
| `u` | p-value of the original's program `score.py` on its full dataset; empty for slot 27 |
| `estimate`, `std_error` | The program's estimate and standard error on the full dataset |
| `commitment_model`, `commitment_null`, `commitment_population`, `commitment_outcome`, `commitment_sample_restrictions` | The original's declared model and null |

## `sessions.jsonl.gz` (64,712 lines)

One line per session attempt: the originals, the replays, and attempts that were stopped or cut by the provider
and rerun under the same id. The agents' answer texts are left out.

| Field | Meaning |
|---|---|
| `session_id` | Opaque id; all attempts of a replay share it |
| `role` | `original` or `replay` |
| `slot` | Original |
| `component` | Replays: `base` or `ext1` |
| `stratum` | Replays: `tail` (the original program's p-value on the subsample is at most u) or `body` |
| `bank_index` | Replays: subsample index (see `data/raw/pathways/SOURCE.md`) |
| `ref_p` | Replays: the original program's p-value on that subsample |
| `file`, `line` | The node's results file and the record's line in it |
| `sink` | `results`, or `limited` for an attempt stopped by a usage limit (not counted) |
| `attempt`, `route`, `effort` | Attempt number; `api` or `openrouter-flex`; `high` |
| `started`, `finished` | Unix times |
| `data_sha256` | SHA-256 of the `data.csv` the session received |
| `has_score_py` | Whether the session left a `score.py` |
| `commitment` | The session's declared model and null (five strings), or null |
| `model_null_match`, `reason` | Replays: the replay's own verdict on whether its declaration matches the original's, and its reason; null if the comparison returned nothing |
| `u`, `score_returncode`, `score_result` | Originals: the program's p-value on the full data, exit code and JSON output |
| `phases` | Per phase (`analysis`, `commitment`, `comparison`): seconds, return code, faults, numbers of events and commands, and token usage |

## `reports.jsonl.gz` (100 lines)

`slot` and `final_report`, the original agent's final answer (first 20,000 characters).

## `programs/slot-SS/`

Each original's `score.py`, the program the agent wrote to rerun its chosen analysis:
`python score.py --data D --output O` writes the estimate, standard error and p-value as JSON.
