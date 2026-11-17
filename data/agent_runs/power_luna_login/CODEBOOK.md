# Cohort `power_luna_login`: power design, GPT-6-Luna on ChatGPT Pro logins

Records of the power design (Table 1, Panel D).

- Agent: GPT-6-Luna, high reasoning effort, Codex CLI 0.157.0, on two ChatGPT Pro subscription logins
  (`subscription-A` and `subscription-B`). The originals ran on login A; the replays ran on A, and from
  5 November also on B, split by run order before any match rate was read. Run on 4 and 5 November 2026.
- Prompts: the open-ended design's research, commitment and comparison prompts and codebook, unchanged
  (`code/02_open_ended/data_generation/prompts/`, `data/raw/pathways/CODEBOOK.md`).
- Data: `data/raw/pathways_power/power-NN.csv`. A replay's dataset is reference subsample `bank_index`
  of its original's dataset: m = 200 rows drawn without replacement in the order
  `random.Random(f"{seed}-bank-{bank_index}").sample(range(2000), 200)`
  (`subsample_indices` in module 02's `cluster/runner.py`), m/n = 0.10. `data_sha256` is the hash of the
  file the session received.
- Sessions: 200 originals; 198 usable (slots 36 and 47: the agent's own program fails on its own data).
  46,808 replay sessions (7,208 tail, 39,600 body), all classified. 14,905 earlier attempts were cut and rerun on the same subsample:
  201 `cut` records, whose turn did not complete (200 of them in a provider outage on 4 November), and
  14,704 `rate_limited` records. Each usable original has 40 tail and 200 body replays, except at +9.6 points
  (odd slots), where 31 banks held fewer than 40 tail subsamples and 3 (slots 29, 71 and 121) none, which
  leaves their bounds vacuous.
- Extraction: `code/04_power/data_generation/extract_records.py` from the run's per-node result files.
  Answer texts, the replays' reports, account names and meter readings are dropped; full transcripts are
  not shipped.

## `sessions.jsonl.gz` (61,913 lines, one per session attempt)

| Field | Description |
|---|---|
| `session_id` | Opaque case id; an original's id is also its program's file name in `programs/` |
| `role` | `original` or `replay` |
| `record` | `result` (the attempt's result record) or `rate_limited` (an attempt cut by a rate limit, written to the run's separate log of limited attempts) |
| `status` | `completed` or `program_error` (originals: whether the saved program returned a p-value on the original's own data); `classified`, `cut` or `rate_limited` (replays) |
| `attempt` | Attempt number recorded by the runner within one launch; it starts again at 1 when a later launch reran the session, so it is not always the number of earlier cut attempts plus one (228 sessions) |
| `route` | `subscription-A` or `subscription-B` |
| `model`, `effort` | `gpt-6-luna`, `high` |
| `started`, `finished` | Unix time (seconds) |
| `data_sha256` | SHA-256 of the session's `data.csv` |
| `phases` | Per turn (`analysis`, `commitment`, `comparison`): `seconds`, `returncode`, `timed_out`, `ok`, `error` (when cut), `n_events`, `n_commands`, `usage` (input, cached input, cache-write input, output and reasoning output tokens) |
| `commitment` | The session's answer to the commitment prompt: `model`, `null`, `population`, `outcome`, `sample_restrictions` |
| `has_score_py` | Whether the session saved a program |
| `slot` | Dataset slot of the original (0 to 199) |
| Originals only: `seed`, `p_value`, `program_result`, `program_returncode`, `final_report` | Generator seed; u, the saved program's p-value on the full dataset; the program's output (`estimate`, `std_error`, `p_value`) and exit code; the agent's final report |
| Replays only: `original`, `stratum`, `bank_index`, `comparison` | The original's session id; `tail` or `body` as assigned when the replays were drawn; the reference subsample; the verdict (`model_null_match`, `reason`) |

The bank p-value of a subsample is in `data/intermediate/04_power/bank_p.csv.gz`.

## `originals.csv` (200 rows)

| Field | Description |
|---|---|
| `session_id`, `slot`, `seed` | As above; seed = 9,000,000 + slot |
| `arm`, `tau` | `moderate` (even slots, tau = 0.28) or `strong` (odd slots, tau = 0.43) |
| `lead_outcome`, `lead_followup` | The one outcome column (without suffix) and follow-up month the saved program reads |
| `lead_family` | `primary` (employment at 12 months), `structural` (hours, earnings or training at 12 months, which move with employment), `unaffected` (every other outcome, true effect zero) |
| `lead_type` | `ate` (all participants), `subgroup` (analyst-imposed subgroup) or `interaction` (contrast between groups) |

The lead codes are the run's frozen record, copied unchanged: `code/04_power/data_generation/code_headlines.py`
wrote them from each program and commitment (no p-value read) before any reference bank was built. No analysis
uses `lead_type`, which codes slots 26, 59, 70, 154 and 174 as `interaction` although they test a single treatment
coefficient; the interaction tests are slots 56, 66, 116 and 156, and module 04 reads them from the commitments.

## `programs/` (200 files)

Each original's saved `score.py`, verbatim, named by session id. Run as
`python score.py --data <csv> --output <json>`; it writes `estimate`, `std_error` and `p_value`.
