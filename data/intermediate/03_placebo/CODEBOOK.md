# Module 03 intermediate data: reference banks of the placebo originals

## `bank_pvalues.csv.gz`

One row per reference subsample of each of the 102 placebo originals (470,000 rows), written by
`code/03_placebo/data_generation/export_cluster_records.py` from the cohort's bank files
(`make_references.py`, run on the cluster). `code/03_placebo/03_replay_counts.py` reads it to rebuild each
original's tail and body strata.

| Field | Content |
|---|---|
| `slot` | The original, 0-101 (study = ("wellness", "teachers", "consumers")[slot mod 3]) |
| `bank_index` | Index j of the subsample in the original's bank, 0 to bank size minus 1. Subsample j holds rows `random.Random(f"{seed}-bank-{j}").sample(range(n), m)` of the original's dataset, in that order, with seed = 9,000,000 + slot, n the dataset's rows and m = round(0.10 n) |
| `p_value` | The original's frozen program (`data/agent_runs/placebo_luna/programs/score_slotNNN.py`) run on subsample j: its reported p-value, written with full precision. Empty for the 1,756 subsamples on which the program failed; those are in neither stratum |

A subsample is in the tail when `p_value` is at most the original's u (`p_value` of the original's record in
`data/agent_runs/placebo_luna/sessions.jsonl.gz`) and in the body otherwise. Bank sizes are 2,000 (67
originals), 4,000 (20), 8,000 (8), 16,000 (2) and 32,000 (5): a bank started at 2,000 and doubled until both
strata could supply their 40 tail and 200 body replays, up to 32,000. Slot 75's bank has no tail subsample.

Degenerate p-values, from fits on very small subsamples, are kept as the programs reported them and count in
their stratum like any other (2,397 values are exactly 1 and one is exactly 0).

Recomputing every bank takes about 200 hours on two cores.
