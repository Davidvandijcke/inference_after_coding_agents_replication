# 01 Fixed-rule design

Table 1 Panel A and the fixed-rule bars of Figure 3(a). On 400 null datasets with six independent N(0, 1) outcomes
and n = 1,000, a programmed selector reports the smallest of the six z-test p-values with probability 0.8 and a
uniformly chosen one otherwise; replays use subsamples of m = 31, with 25, 100, 400 and 1,000 replays per stratum.
The exact selective p-value is known, so the corrected p-value can be compared with it. No model calls.

## Analysis (run by `run_all.sh`, under a second)

| Script | Writes |
|---|---|
| `01_replay_estimates.R` | `data/derived/01_fixed_design/fixed_rejections.csv`: corrected and exact p-values and the 5% decisions per dataset and budget (module 02 reads it for Figure 3(a)) |
| `02_panel_a.py` | `paper/tabs/simulation_panel_a.tex`, `results/numbers/01_fixed_design.json` |

Input: `data/intermediate/01_fixed_design/replay_draws.csv`, the Monte Carlo draws (selected test, its p-value,
tail and body matches and replay intervals for each dataset and budget; see its `CODEBOOK.md`).

## Monte Carlo (`data_generation/simulate.R`, not rerun)

`Rscript code/01_fixed_design/data_generation/simulate.R 1 400 replay_draws.csv 2` regenerates the input exactly
(about 5,800 CPU-seconds), with R's default generator and seeds 2026102200 + offset + dataset (one offset per
random step), and 2026172200 + dataset and 2026182200 + dataset for the subsamples added where a bank of 50,000
held fewer than 1,000 tail members.
