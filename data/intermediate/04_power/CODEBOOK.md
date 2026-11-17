# Intermediate results of module 04 (power design)

Saved outputs of computations that ran on the cluster or take long; `code/04_power/run.sh` reads them.

## `bank_p.csv.gz` (8,226,000 rows)

Each usable original's saved program run on the reference subsamples of its dataset (m = 200 of n = 2,000
rows, without replacement), scored on the cluster by `code/04_power/data_generation/bank_chunks.py`:
subsamples 0 to 3,999 of each bank by the runner's standard scorer, one sandbox per subsample (706,000 rows),
and the rest by the pre-loaded scorer `fast_guest.py` (7,520,000 rows), which reproduced the standard scorer
exactly on the first 2,000 subsamples of every program. The bank starts at 2,000 subsamples and doubles until
it holds 40 tail and 200 body subsamples, up to 128,000 (`make_references.py`). `extract_records.py` wrote this
file from the bank files.

| Field | Description |
|---|---|
| `slot` | Dataset slot of the original (0 to 199; slots 36 and 47 have no bank) |
| `bank_index` | Subsample j; its rows are `random.Random(f"{seed}-bank-{j}").sample(range(2000), 200)` with seed = 9,000,000 + slot |
| `p_value` | The program's p-value on that subsample (every subsample returned one) |

A subsample is in the tail stratum when `p_value` is at most the original's p-value u, otherwise in the body.

## `program_draws.csv.gz` (1,584,000 rows)

Each usable original's saved program run on new datasets of 2,000 participants from the generator. Both
kinds of draws were made and scored by `code/04_power/data_generation/bank_chunks.py settle`, with the
pre-loaded scorer `fast_guest.py`; `extract_records.py` wrote this file from the run's draw files.

| Field | Description |
|---|---|
| `slot` | Dataset slot of the original |
| `effect` | `yes`: drawn with the original's own effect (tau of its arm), seed 8,000,000,000 + slot x 1,000,000 + j; `no`: the same seeds with tau = 0 |
| `j` | Draw index, 0 to 3,999 |
| `p_value` | The program's p-value on that dataset (every draw returned one) |

## `effect_calibration.json`

Written by `code/04_power/data_generation/calibrate_effect.py` (no model calls).

| Key | Description |
|---|---|
| `big_n` | Size of the one large generator draw used for population quantities (4,000,000) |
| `reps` | Monte Carlo replications per arm (20,000) |
| `exact_solutions` | Latent shifts at which the analytic power of the pre-specified test at n = 2,000 is 80% and 99% (keys `80`, `99`); the design rounds them to 0.28 and 0.43 |
| `arms` | One entry per arm (`null`, `moderate`, `strong`) |
| `arms.<arm>.tau` | Latent shift of employment at 12 months |
| `arms.<arm>.population.<outcome>` | For employment, hours, earnings and training at 12 months: `ate` (population effect), `mean_control`, `sd`, `within_var` (mean within-site, within-arm variance) |
| `arms.<arm>.population.employed_m12_ate_by_group` | Population effect on employment at 12 months by baseline subgroup |
| `arms.<arm>.analytic.<n>` | At n = 2,000 and m = 200: noncentrality `ncp` and analytic power `power_05`, `power_01` of the pre-specified test |
| `arms.<arm>.monte_carlo.<outcome>\|<group>` | Monte Carlo rejection rates (`reject_05`, `reject_01`), simulation SE `mc_se_05` and `median_p` of the pre-specified test by outcome and subgroup; `sub\|...` entries are on subsamples of 200 |
