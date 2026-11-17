# Miao, Pritchard and Zou (2026)

Citation: Miao, Jiacheng, Jonathan K. Pritchard and James Zou (2026), "The Agentic Garden of Forking Paths", arXiv
preprint arXiv:2607.01507 (version 1). Replication package: https://github.com/jmiao24/AgenticBootstrap, commit
`7c8925f9663e34b09e14e9c14ad67a6c3b547bb9` (27 August 2026). Accessed 14 October 2026 (repository) and
4 November 2026 (paper).

| File | Source | License |
|---|---|---|
| `paper/supplementary_figure_3_page35.svg` | Page 35 of https://arxiv.org/pdf/2607.01507v1 (Supplementary Figure 3), converted with `pdftocairo -svg -f 35 -l 35` (Poppler 25.05.0); SHA-256 `7a1ef4c92761bb16959e144008325d010688db15665bd24131061cdad26c7704` | CC BY 4.0 (the arXiv record's license) |
| `LICENSE` | The repository's MIT license, for the modified copies of its `codes/Round1_reviewer.mjs` and `prompts/data/issp.md` in `code/05_immigration_personas/data_generation/` (`repair_review/r1_repair.mjs`, `analysts/prompts/issp_paper.md`) | MIT, Copyright (c) 2026 Jiacheng Miao |

`code/05_immigration_personas/01_read_sfig3c.py` reads the z-statistics plotted in panel c of the page.

The persona runs of module 05 used files of the repository at that commit, which are not copied here: the persona
prompts (`prompts/persona/issp/`, used verbatim as the analysts' system prompt), the task prompt and its assembly
(`prompts/instructions/`, `prompts/data/issp.md`, `prompts/load_prompts.mjs`), the agent harness
(`codes/agentic_bootstrap.mjs`, which `analysts/run_miao.mjs` reimplements) and the inference-repair step
(`codes/Round1_reviewer.mjs`, `codes/strip_py.py`). The review prompt and the two example specification summaries of
the paper's Appendix B (CC BY 4.0) are in `code/05_immigration_personas/data_generation/repair_review/prompts/`. The
repository's ISSP file is respondent-level ISSP data; see `data/raw/issp/SOURCE.md`.
