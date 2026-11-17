# 09 Card-Krueger with other coding agents

Table 8 and its in-text numbers. The Card-Krueger application of module 08 is rerun with four other agent cohorts,
each with its own original on all 410 stores and 3,000 replays on reference subsets of 150 stores from module 08's
bank. The table's first row is module 08's GPT-6-Luna cohort; no cohort is pooled with another.

## Analysis (run by `run_all.sh`, about 1 minute)

| Script | Writes |
|---|---|
| `01_counts.py` | `data/derived/09_card_agents/counts.csv`: matches and classified replays by stratum; checks each replay's stratum and subset p-value against its original's bank |
| `02_score_bank.R` | `*_bank_p_rescored.csv`: each original's program rerun on its replays' subsets and checked against the saved bank (with `all`, on all 20,000 subsets in about 2 minutes) |
| `03_replay.R` | `estimates.csv`: corrected p-value, multiplier and one-sided 95% bounds per cohort |
| `04_keyword_rates.py` | `keyword_rates.json`: decline and naming rates (rule in `declines.py`) |
| `05_mask_packet.py` | `masked_packet/`: the masked packet rebuilt from the unmasked one and checked against the one shipped |
| `06_table.py` | `paper/tabs/card_agents.tex`, `results/numbers/09_card_agents.json` |

Inputs: `data/agent_runs/card_terra/`, `card_sol/`, `card_sonnet/`, `card_sonnet_masked/` (see each `CODEBOOK.md`),
their originals' banks in `data/intermediate/09_card_agents/`, and module 08's packet, subsets, and GPT-6-Luna records,
bank and streams. R needs jsonlite and sandwich. The records are author-generated (CC BY 4.0), except the masked
`packet/data.csv`, which holds Card and Krueger's records under neutral names and keeps the status of
`data/raw/card_krueger/`.

## Code that produced the agent records (`data_generation/`, not rerun)

1. Originals, each given module 08's packet or the masked one as `inputs/`: `terra/pilot.py original`,
   `sol/pilot.py original`, `MMO_CLAUDE_BIN=<Claude Code CLI> sonnet/card_claude.py original --account a`.
2. Banks: `terra/score_bank.R`; Sol and Sonnet ran it with their own file names (Sonnet's in its sandbox, offline).
3. Replay lists: `terra/make_main_roster.R`, `sol/make_main_roster.R`, `sonnet/make_roster.R`; for the masked cohort
   `sonnet_masked/make_pilot_roster.R` (200 replays), `make_ext_roster.R` (600) and `make_ext2_roster.R` (2,200).
4. Terra and Sol replays: `terra/main.py` (laptop), then on the cluster `gl_task.py --tasks 4 --account
   <login>=<run>/auth/<login>:10` (Sol with `PENDING=<list>`), with `main.py` and `pilot.py` beside it. Sol ran
   `terra/main.py` with its own `pilot.py`; only the cohort's name and the subscription-meter stop `PAUSE_AT` (97
   from 1 November, 100 from 3 November) differed.
5. Sonnet replays: `sonnet/card_claude.py split --accounts a --name L1`, then `replays --account a --pending
   pending-L1-a.json`, with `common.py`, `run_claude.py` and `clean_bash.sh` of module 10's `four_agent_null_run/`
   beside it; the masked cohort ran the same file.
6. Review packets: Terra `summarize_main.py`, `review_packets.py`, `rereview_packets.py` (adds the data facts); Sol
   `sol_rereview_packets.py`, `sol_rereview_packets_more.py`; Sonnet `review_packets.py`; masked `review_packets.py`
   and `review_packets_ext.py`.
7. Review: every cohort ran `terra/rereview_gl.py --tasks <n> --workers <k>`, with its own logins and meter stop.
8. Labels: Terra `review_join.py` (case list), then `rereview_join.py` (also module 08's Luna labels); Sol
   `sol_rereview_join.py`, which Sonnet ran with its own paths; masked `review_join_full.py`.
9. Blind reads of Sol's first 831 reports (`report_reads.csv`): `sol/build_review_card.py` (with
   `card_correction_scan.py`, rule `review_rule_facts_card.txt`), read by module 10's `four_agent_null_run/review.py`
   (schema `facts_card`), joined by `sol/export_card_review.py`; both ran under that run's `analysis.sbatch` and
   import its `common.py`.
10. `keyword_scans/`: decline and naming scans of the Sol and Sonnet session archives, with `../declines.py`.
11. `extract_records.py` and `extract_cluster_records.py` wrote the shipped records from the run folders.

Earlier versions, job files and helper scripts of the runs are not included and are available from the authors, as are
the session archives and the pilots (Terra 40 replays, Sol 20), which set the tail and body split and enter no result.
The command that compacted the unmasked Sonnet records for its review scripts was typed inline and never saved.

Paths, accounts and private folder names are replaced by placeholders, and three comments that quoted the authors'
instructions to the agents or named a synced folder are shortened; the code is otherwise as run. Module 08 also
imports `terra/pilot.py`. All cohorts ran at high reasoning effort on subscription logins, with no dollars charged:

| Cohort | Harness and route | Dates (2026) | Replays (tail + body) | Classified |
|---|---|---|---|---|
| GPT-6.1-Terra | Codex CLI 0.164.0-alpha.11.1 on a laptop (368 replays), 0.165.0 on the cluster (2,632); two ChatGPT subscription logins | 29 Oct; review 30 Oct | 3,000 (1,081 + 1,919) | 2,982 |
| GPT-6.1-Sol | Codex CLI 0.165.0 on the cluster; two ChatGPT Pro logins | 29 Oct to 3 Nov; review 31 Oct to 3 Nov | 3,000 (1,500 + 1,500) | 2,986 |
| Claude Sonnet 5.5 | Claude Code CLI 2.1.314 on the cluster; one Claude login | 3 to 4 Nov; review 5 to 6 Nov | 3,000 (1,500 + 1,500) | 2,988 |
| Sonnet 5.5, masked packet | the same; two Claude logins | 4 to 5 Nov; review 5 to 6 Nov | 3,000 (1,500 + 1,500) | 2,977 |

In every cohort a match is the label of a review by GPT-6.1-Sol at high effort, blind to stratum and verdict, which
compared each replay's committed declaration with the original's, given a paragraph of checked facts about the data
(`review_prompt.json`). Unresolved pairs and the replays that did not finish (5 Terra, 10 Sol; never retried) are
excluded. The masked cohort began with 200 replays and was extended to 3,000 after their result had been seen. The
keyword scans cover different text: Sonnet's the analysis turn's thinking, messages and tool calls; Sol's all three
turns with tool output; Luna's the analysis turn's messages, commands and output. Terra's replays shared a `/tmp`: 69
listed or read a file another replay had left there, and 10 of them read or ran it; the first 263 laptop replays also
ran before the sandbox denied reads outside the run folder (`tmp_exposed` and `read_isolation` in
`card_terra/replays.csv.gz`, beside the run's other environment defects). Sol and Sonnet replays had private `/tmp`s.
