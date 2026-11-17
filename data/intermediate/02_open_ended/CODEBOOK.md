# Open-ended design: reference-bank p-values

`bank_pvalues.csv.gz` (1,616,000 rows, 99 banks): for each usable original, the p-value its program
(`data/agent_runs/open_ended_luna/programs/slot-SS/score.py`) returned on every subsample of its reference bank. A
subsample is in the tail when its p-value is at most the original's `u`. Built by
`code/02_open_ended/data_generation/cluster/make_references.py` and `extend_references.py`. Author-generated, CC BY 4.0.

| Field | Meaning |
|---|---|
| `slot` | Original |
| `bank_index` | Subsample index j: the 200 rows of `data/raw/pathways/main-SS.csv` listed by `random.Random(f"{8_000_000 + SS}-bank-{j}").sample(range(2000), 200)` |
| `p_value` | The program's p-value on that subsample; empty where the program failed (17,075 subsamples, never drawn as replays) |
