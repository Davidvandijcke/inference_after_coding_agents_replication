# Miao, Pritchard and Zou (2026): paper page and repository files

Citation: Miao, Jiacheng, Jonathan K. Pritchard and James Zou (2026), "The Agentic Garden of Forking Paths", arXiv
preprint arXiv:2607.01507 (version 1). Replication package: https://github.com/jmiao24/AgenticBootstrap, commit
`7c8925f9663e34b09e14e9c14ad67a6c3b547bb9` (27 August 2026). Accessed 14 October 2026 (repository) and
4 November 2026 (paper). Each of the eleven files under `repository/` equals its blob at that commit.

## Files

| File | Source | License |
|---|---|---|
| `paper/supplementary_figure_3_page35.svg` | Page 35 of https://arxiv.org/pdf/2607.01507v1 (Supplementary Figure 3), converted with `pdftocairo -svg -f 35 -l 35` (Poppler 25.05.0) | CC BY 4.0 (the arXiv record's license) |
| `repository/LICENSE` | Repository root | MIT, Copyright (c) 2026 Jiacheng Miao |
| `repository/codes/agentic_bootstrap.mjs` | The authors' agent harness (Claude Agent SDK), which `code/05_immigration_personas/data_generation/analysts/run_miao.mjs` reimplements | MIT |
| `repository/codes/Round1_reviewer.mjs`, `repository/codes/strip_py.py` | The authors' inference-repair step, run with the changes marked `ALIGN:` in `code/05_immigration_personas/data_generation/repair_review/r1_repair.mjs` | MIT |
| `repository/prompts/load_prompts.mjs`, `repository/prompts/data/issp.md`, `repository/prompts/instructions/10round_7axis.md`, `repository/prompts/instructions/_round_template.md` | The authors' task prompt and its assembly | MIT |
| `repository/prompts/persona/issp/anti_immigration.md`, `pro_immigration.md`, `neutral.md` | The persona prompts, used verbatim as the analysts' system prompt | MIT |

All files are unmodified. `Round1_reviewer.mjs` contains the authors' own home folder in a sandbox rule.

| File | SHA-256 |
|---|---|
| `paper/supplementary_figure_3_page35.svg` | `7a1ef4c92761bb16959e144008325d010688db15665bd24131061cdad26c7704` |
| `repository/LICENSE` | `edbacdf5f62caac295233fe0746dd314f1e91bd4cf31e051a7295ebd439bb497` |
| `repository/codes/agentic_bootstrap.mjs` | `3922123c975cd5ef0f9a53f080267009505fa1bd89ff3ccd039c259503d6235c` |
| `repository/codes/Round1_reviewer.mjs` | `cb91e138c3951bee7557fef745fcb5d50ca58d228ecddd75aa1ced44f47c5644` |
| `repository/codes/strip_py.py` | `7a3d47fff396d7ffbe2263f4839dedb3759b9bc982002cf60fab986ca06e96cb` |
| `repository/prompts/load_prompts.mjs` | `b570016d5f3593398ee374016cf59a863e1fb6be1bd77aac383259c3924ac233` |
| `repository/prompts/data/issp.md` | `f8713971f486e03a87f2d62014f44a4e03dd7e7c9337a34b76739dcf604943b3` |
| `repository/prompts/instructions/10round_7axis.md` | `953e12e5e9215da51496174ad493d05a15857b80b8e4bca3838af0b1108b9699` |
| `repository/prompts/instructions/_round_template.md` | `9b38099f2cfa0a0bc3a1f7fc136359fa896a1adbfc11120054affc98986caf58` |
| `repository/prompts/persona/issp/anti_immigration.md` | `3e0b5c66539774f4824188a01160696bc58cff33e6efb7f4c1985581effc0496` |
| `repository/prompts/persona/issp/pro_immigration.md` | `96d790d37d0b4bb62d2b84c448366ccb64887331a2ff6ac71b8db26851f8da05` |
| `repository/prompts/persona/issp/neutral.md` | `835eb85b28a29af9069aa9a55896736d7b01c6c3b70c97903c6e33c3c22f2874` |

`code/05_immigration_personas/01_read_sfig3c.py` reads the z-statistics plotted in panel c of the page. The review
prompt and the two example specification summaries of the paper's Appendix B, used verbatim (CC BY 4.0), are in
`code/05_immigration_personas/data_generation/repair_review/prompts/`.

Not shipped: the repository's ISSP file `examples/issp/data/data_clean.csv` (respondent-level ISSP data; see
`data/raw/issp/SOURCE.md`).
