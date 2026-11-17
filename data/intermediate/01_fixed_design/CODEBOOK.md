# Fixed-rule design: Monte Carlo draws

`replay_draws.csv` (1,600 rows: 400 datasets times 4 replay budgets) holds the output of
`Rscript code/01_fixed_design/data_generation/simulate.R 1 400 replay_draws.csv`, which reproduces
it exactly from fixed seeds. Author-generated, CC BY 4.0. Real numbers are written with 17
significant digits.

| Field | Meaning |
|---|---|
| `case_id` | Dataset, 1-400 |
| `b` | Replays per stratum: 25, 100, 400 or 1,000. The budgets are nested: the first `b` replays of the same random ordering of each stratum |
| `n`, `m`, `k`, `epsilon` | The design: observations per dataset (1,000), subsample size (31), number of outcomes and z-tests (6), and the selector's probability of reporting a uniformly chosen test instead of the smallest p-value (0.2) |
| `q` | The test the selector reported on the full dataset (1-6) |
| `u` | Its two-sided p-value on the full dataset |
| `x1`, `b1` | Tail (subsamples whose p-value for test `q` is at most `u`): replays on which the selector chose `q` again, and replays (`b1` = `b`) |
| `x0`, `b0` | The same for the body (p-value above `u`) |
| `lower95`, `upper95` | Two-sided 95% signed-LR interval for the corrected p-value (`replay_interval_lr`, gamma = 0.05) |
| `upper_one95` | One-sided 95% upper bound (upper end of the two-sided 90% interval, gamma = 0.10); the replay test rejects at 5% when it is below 0.05 |
