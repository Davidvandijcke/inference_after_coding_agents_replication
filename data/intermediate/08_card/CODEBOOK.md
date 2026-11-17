# 08_card: the Card-Krueger reference bank

The 20,000 reference subsets of the Card-Krueger application and the original program's p-value on each. Modules 08
and 09 read these files.

Each subset holds 150 of the 410 stores in `data/agent_runs/card_luna/packet/data.csv` (whole stores, both waves,
m/n = 150/410), drawn without replacement with R's L'Ecuyer-CMRG generator from seed 2026101731, one stream for all
20,000 draws in order. A subset's p-value is that of GPT-6-Luna's emitted program
(`data/agent_runs/card_luna/programs/original-analysis.R`) on the subset's rows. Subsets with p at most the original's
full-data p-value u = 0.0252 form the tail (2,395), the others the body (17,605). The 3,000 replays used the first
1,142 tail subsets and the first 1,858 body subsets of two seeded permutations (seeds 2026101732 and 2026101733); each
replay's subset is `bank_id` in `data/agent_runs/card_luna/replays.csv.gz`, and its `data.csv` held the subset's rows
in draw order. The other Card cohorts of module 09 replayed on subsets of the same bank.

| File | Fields |
|---|---|
| `card-bank-subsets.csv.gz` | `bank_id` (1 to 20,000, in draw order); `rows`: the subset's 150 row numbers in `data.csv` (1-based, first data row = 1), separated by spaces, in draw order. |
| `card-bank-p.csv` | `bank_id`; `p_value`: the original program's p-value on the subset, as computed on the cluster in October 2026 (R 4.5.1, Linux), to 17 significant digits. |

`code/08_card/data_generation/build_bank.R` regenerates both files, and `code/08_card/02_bank.R` checks them against a
fresh draw and a full rescoring on every run.
