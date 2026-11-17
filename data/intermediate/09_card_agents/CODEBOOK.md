# data/intermediate/09_card_agents: reference banks of the four other-agent originals

Each file is one original's emitted program (`data/agent_runs/card_<cohort>/programs/original-analysis.R`) run
unchanged on each of the 20,000 reference subsets of 150 whole stores (`data/intermediate/08_card/card-bank-subsets.csv.gz`,
module 08). A subset is in the tail when its p_value is at most the original's full-data p-value u
(`original.json`, `p_value`) and in the body otherwise.

| File | Original | u | Tail subsets |
|---|---|---|---|
| `terra_bank_p.csv` | GPT-6.1-Terra | 0.0284320863889777 | 2,287 |
| `sol_bank_p.csv` | GPT-6.1-Sol | 0.0269251960652278 | 2,236 |
| `sonnet_bank_p.csv` | Claude Sonnet 5.5 | 0.0411782718890429 | 2,929 |
| `sonnet_masked_bank_p.csv` | Claude Sonnet 5.5, masked packet (same rows of the masked `data.csv`) | 0.0404878671868477 | 3,004 |

| Field | Meaning |
|---|---|
| bank_id | Id of the reference subset (1 to 20,000), as in module 08's subsets file. |
| p_value | The program's p-value on that subset (none failed). |
| estimate | The program's estimate on that subset. |

The values of u are as recorded in each `original.json` (15 significant digits), and the files print 15
significant digits.

`code/09_card_agents/02_score_bank.R all` regenerates all four into `data/derived/09_card_agents/` and checks them
against these files, every p-value and estimate within 1e-12. On a Mac the Terra and Sol banks come out
byte-identical; the two Sonnet banks, scored on the cluster, differ in the last printed digit, with identical tail sets.
