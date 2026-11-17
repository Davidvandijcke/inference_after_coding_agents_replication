# Cohort `placebo_luna`: real-data placebo design

Agent records of Table 1, Panel C. GPT-6-Luna (high reasoning effort, Codex CLI 0.157.0, signed in with two
ChatGPT Pro logins) analyzed 102 placebo datasets built from three published experiments (34 per study),
then was replayed on 40 tail and 200 body subsamples of each dataset at m/n = 0.10 (24,440 replays). Slots
0-50 ran on 30 October 2026 and slots 51-101 on 31 October 2026. Banks and replays ran only after a gate fixed
before any original ran had passed: at least 7 of the first 51 originals significant at 5%. The research prompt
and codebooks are in `code/03_placebo/data_generation/runtime/`, and each session received its study's
`codebooks/<study>.md` as `CODEBOOK.md`; the commitment and comparison prompts are those of
`code/02_open_ended/data_generation/prompts/`. `code/03_placebo/data_generation/export_cluster_records.py`
extracted these files from the cohort's run folder. All values were written by the agents or by the cohort's
runner; nothing was relabeled.

## `sessions.jsonl.gz`: one record per session (102 originals and 24,440 replays)

Each record is the session's final, classified attempt. A session the provider cut off was restarted from
scratch with the same data (6,085 cut attempts on 4,461 sessions); cut attempts are counted, not shipped.

| Field | Content |
|---|---|
| `session_id` | Opaque session identifier (32 hex characters) |
| `role` | `original` or `replay` |
| `cohort` | `placebo_luna` |
| `slot` | 0-101, the original (for a replay, the original it replays) |
| `study` | `wellness`, `teachers` or `consumers`, equal to ("wellness", "teachers", "consumers")[slot mod 3] |
| `model`, `effort`, `harness` | `gpt-6-luna`, `high`, `codex-cli 0.157.0` |
| `route` | `subscription:primary` or `subscription:second`: which of the two ChatGPT Pro logins ran the session |
| `attempt` | Attempt number of this record within one launch (1 unless earlier attempts of that launch were cut). A session rerun by a later launch starts again at 1, so `attempt` is not always `attempts_cut` + 1 (81 replays) |
| `attempts_cut` | Number of earlier attempts of this session that the provider cut off |
| `started`, `finished` | Unix times of the recorded attempt |
| `data_sha256` | SHA-256 of the `data.csv` the session received |
| `status` | `completed` (originals: a program and a commitment) or `classified` (replays: a comparison verdict). All 24,542 records have it |
| `phases` | Per phase (`analysis`, `commitment`, and for replays `comparison`): `seconds`, `returncode`, `timed_out`, `ok`, `n_events`, `n_commands` and token `usage` (`input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, `output_tokens`, `reasoning_output_tokens`) |
| `commitment` | The session's answer to the commitment prompt: `model` (statistical model), `null`, `outcome`, `population`, `sample_restrictions` |
| `final_report` | The session's final message to the user, as written |

Originals only:

| Field | Content |
|---|---|
| `seed`, `n` | Seed of the placebo dataset (9,000,000 + slot) and its number of rows |
| `has_score_py` | Whether the session saved its frozen program (true for all 102) |
| `p_value` | u, the frozen program's p-value on the full dataset |
| `score` | The program's output on the full dataset: `estimate`, `std_error`, `p_value` |

Replays only:

| Field | Content |
|---|---|
| `original_session_id` | `session_id` of the original replayed |
| `stratum` | `tail` (the original's program gives p at most u on this subsample) or `body` |
| `bin` | The equal-width p-value bin of the subsample within its stratum (4 tail bins on (0, u], 10 body bins on (u, 1]), recorded only: the replayed subsamples were drawn uniformly without replacement within each stratum, not by bin |
| `bank_index` | Index j of the subsample in the original's reference bank: rows `random.Random(f"{seed}-bank-{j}").sample(range(n), m)` of the original dataset, m = round(0.10 n) |
| `bank_p_value` | The original program's p-value on that subsample |
| `own_match` | The replay's own verdict that it committed to the original's model and null (`model_null_match`) |
| `comparison_reason` | The replay's explanation of that verdict |

Slot 75's program reports u = 1.4e-11; no subsample in its 32,000-subsample bank reaches it, so it has 200
body replays and no tail replays.

## `programs/score_slotNNN.py`

Each original's frozen program, as saved in its workspace: `python score_slotNNN.py --data <csv> --output
<json>` reruns the chosen analysis and writes `estimate`, `std_error` and `p_value`. The cohort ran them with
numpy 2.5.4, pandas 3.0.6, patsy 1.0.3, scipy 1.18.1 and statsmodels 0.15.0.

## `recorded_sha256.csv`

The SHA-256 of the four base files held by the agents' runtime, recorded when the cohort ran.
`code/03_placebo/02_make_placebo_datasets.py` checks the rebuilt base files against it, and every dataset
and subsample against `data_sha256` in `sessions.jsonl.gz`.

| Field | Content |
|---|---|
| `file` | Path under `data/derived/03_placebo/`: `base/wellness_base.csv`, `base/teachers_base.csv`, `base/consumers_base.csv`, `base/experiments.json` |
| `sha256` | SHA-256 of that file, hexadecimal |

## Reference banks

See `data/intermediate/03_placebo/CODEBOOK.md`.
