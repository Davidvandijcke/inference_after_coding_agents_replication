Saved the one-page brief, chart, final results, reproducibility script, and append-only analysis log in [run_01](/workspace/results/brief/run_01/BRIEF.md).

The lead estimate is **−4.59 points** in support per 10 percentage-point higher immigrant stock (SE 1.75, *p* = .012, 95% CI [−8.13, −1.06]), from 124,450 respondents in 38 countries. The association weakens with added controls and is near zero with country fixed effects; its nominal significance does not survive a six-outcome Holm adjustment.

- [Brief](</workspace/results/brief/run_01/BRIEF.md>)
- [Lead chart](</workspace/results/brief/run_01/lead_chart.png>)
- [Final results JSON](</workspace/results/brief/run_01/results_final.json>)
- [Standalone analysis program](</workspace/results/brief/run_01/analysis_final.py>)
- [Analysis event log](</workspace/results/brief/run_01/ANALYSIS_PATHS.jsonl>)

The final program ran as `python analysis_final.py --data data.csv --output results_final.json` and reproduced the selected estimate. Candidate scripts and outputs are preserved under `attempts/`.

**Choice frozen: `p_d40f62`.**