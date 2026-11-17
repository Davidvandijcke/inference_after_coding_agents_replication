# 06 Realistic request in the immigration application

Figure 4 and Table 3. A coding agent writes an evidence brief for a team that argues for open immigration from the
ISSP Role of Government file (154,921 respondents in 46 countries) and reports a p-value; replay agents repeat the
task on subsamples of 16 whole countries (m/n = 0.35). Sixty more full-data sessions give the reference distribution
of Miao et al.'s m-value.

## Analysis (run by `run_all.sh`, about 25 seconds)

| Script | Writes |
|---|---|
| `01_correction.R` | `data/derived/06_immigration_request/correction.csv`: matches, corrected p-value, multiplier and intervals; checks each replay's stratum against the bank |
| `02_figure.py` | `paper/figs/issp_selection.pdf` |
| `03_mvalue.py` | `mvalue.json`: the m-value, the reference sessions' leads and the clustering of their plans |
| `04_variables_table.py` | `paper/tabs/issp_variables.tex` |
| `05_numbers.py` | `results/numbers/06_immigration_request.json` |

Inputs: `data/agent_runs/issp_request_luna/` and `issp_mvalue_luna/` (session records; see their `CODEBOOK.md`) and
`data/intermediate/06_immigration_request/` (the bank's p-values, the original program's output on the full file and
respondents per country-wave, all computed from the restricted ISSP file; see its `CODEBOOK.md`).

## Code that produced the agent records (`data_generation/`, not rerun)

On the ISSP file that `code/05_immigration_personas/data_generation/build_issp_file.py` builds, with the files of
`data/agent_runs/issp_request_luna/inputs/` copied into the runner's folder and `EXP_ROOT` naming the run folder:

1. `agents/runner.py cases --list <cases> --task <i> --tasks <k> --cap <dollars> --campaign <name> --route api` on
   each node runs sessions in a sandbox (analysis, commitment, comparison); first the three full-data originals.
2. `agents/make_replays.py --originals-list originals.jsonl --tail 500 --body 500 --bank 4000 --max-bank 16000`: the
   bank and the replay list for step 1 (the rerun job got the same list and skipped completed replays).
3. `agents/reference/make_cases.py <issp_clean.csv>`, then `agents/reference/runner.py cases` with the prompts of
   `data/agent_runs/issp_mvalue_luna/inputs/` added: the 60 reference sessions.
4. `review/build_reviewed.py <run folder>`: the list of wording-only matches of the second reading.
5. `extract_request_records.py <research repository>`: wrote both cohort folders from the run records.
6. `score_bank.py`, `count_country_waves.py`: the files of `data/intermediate/06_immigration_request/` from the ISSP
   file (the bank takes about 2 hours on one core).

Cohort: GPT-6-Luna at high effort through Codex CLI 0.157 in a network-restricted sandbox on a Linux cluster, on the
OpenAI API through a cost-capped relay. `issp_request_luna` (27-28 October 2026, $52.85): three full-data originals,
of which the first is corrected (all three led with the same analysis), and 1,000 replays, all completed and
classified; after the account ran out of credit, a rerun job ran 464 replays from scratch on the same subsamples.
`issp_mvalue_luna` (29 October 2026): 60 full-data sessions, analysis phase only, 20 each with the original request,
a request from a team that argues for lower immigration and a nonpartisan request.

The match counts behind Figure 4, 79 of 500 tail and 32 of 500 body replays, are the replays' own verdicts (31 and 7)
plus 73 of the 962 nonmatches that a second reading counted as matches because their stated differences were wording
only. A Claude Code session (Claude Opus 5.5) did that reading on 28 October 2026: it read the stated reasons of 90
nonmatches (86 picked by keyword rules and 4 others), not blind to stratum, and left the rest unread.

The m-value is m = min(1, 2 min(share of analyses with z at or below the original's, share at or above)) over the 811
estimated analyses of the reference logs, without the results that a logged correction names but does not replace,
with each z compared exactly with the original's.

Not included: the API relay; the replays' reports and briefs and the session archives; and, never saved, the command
that wrote `originals.jsonl` (three full-file cases with random identifiers) and the second reading's keyword rules.
The originals, the bank job and the first replay jobs ran an earlier `runner.py`; the shipped one, the rerun's, fixes
the retry identifier the relay rejected. Earlier versions, job files and helper scripts of the runs are not included
and are available from the authors.
