# 09 Card-Krueger with other coding agents

Table 8 and its in-text numbers. The Card-Krueger application of module 08 is rerun with four other agent cohorts,
each with its own original on all 410 stores and 3,000 replays on reference subsets of 150 stores from module 08's
bank. The table's first row is module 08's GPT-6-Luna cohort; no cohort is pooled with another.

## Analysis

`run.sh` takes about 1 minute.

| Script | Writes |
|---|---|
| `01_counts.py` | `data/derived/09_card_agents/counts.csv`: matches and classified replays by stratum, checked against each original's bank |
| `02_score_bank.R` | `*_bank_p_rescored.csv`: each original's program rerun on its replays' subsets, checked against the saved bank |
| `03_replay.R` | `estimates.csv`: corrected p-value, multiplier and one-sided 95% bounds per cohort |
| `04_keyword_rates.py` | `keyword_rates.json`: decline and naming rates (rule in `declines.py`) |
| `05_mask_packet.py` | `masked_packet/`: the masked packet rebuilt from the unmasked one, checked against the shipped copy |
| `06_table.py` | `paper/tabs/card_agents.tex`, `results/numbers/09_card_agents.json` |

Inputs: `data/agent_runs/card_terra/`, `card_sol/`, `card_sonnet/` and `card_sonnet_masked/`, the originals' banks
in `data/intermediate/09_card_agents/`, and module 08's packet, subsets, and GPT-6-Luna records, bank and streams.
A match is GPT-6.1-Sol's label, blind to stratum and verdict, of a replay's declaration against the original's.

## Code that produced the agent records (`data_generation/`, not rerun)

1. Originals: `terra/pilot.py` (the Codex CLI session runner) and `sonnet/card_claude.py`
   (Claude Code, with module 10's `four_agent_null_run/common.py`, `run_claude.py` and `clean_bash.sh`).
2. `terra/score_bank.R` scores the original's program on module 08's 20,000 subsets and `terra/make_main_roster.R`
   draws the tail and body replays; the other cohorts ran versions adapted to their own banks and splits.
3. Replays: `terra/main.py` (laptop) and `terra/gl_task.py` (cluster), running `pilot.py`, and `card_claude.py`; Sol
   ran versions of `pilot.py` and `gl_task.py` adapted to its model and sandbox.
4. `terra/rereview_packets.py` adds the data facts to an earlier review's blinded batches, `terra/rereview_gl.py`
   runs the reviewer and `terra/rereview_join.py` maps labels to cases; the other cohorts ran versions of all three.
5. `sol/build_review_card.py` puts each Sol report into the rule `sol/review_rule_facts_card.txt`; module 10's
   `four_agent_null_run/review.py` read them for `card_sol/report_reads.csv`.
6. `extract_records.py` and `extract_cluster_records.py` wrote the shipped records; the second applies the decline
   rule (`declines.py`) and the "krueger" pattern to every session archive, as the cluster scans did.

The agents are GPT-6.1-Terra, GPT-6.1-Sol and Claude Sonnet 5.5 at high reasoning effort; each cohort is described in
its `CODEBOOK.md`. Rerunning the sessions needs model access and gives new cohorts, since agents sample their outputs.
