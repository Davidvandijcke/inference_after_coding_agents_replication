# 11 Prompt displays

`paper/figs/prompts.pdf`: the agent prompts quoted in Appendix D.1, one grey box per page, included in the paper
page by page with `\promptbox{N}`. `run.sh` takes about 3 seconds and needs latexmk with pdfLaTeX, Ghostscript and
pdftotext.

| File | Does |
|---|---|
| `prompts.tex` | The 22 displays, with omissions marked [...] |
| `01_check_displays.py` | Checks every display against the prompt file its cohort received and stops on a mismatch |
| `run.sh` | Runs the check, compiles `prompts.tex` and converts all glyphs to outlines |

| Page | Display | File in the package |
|---|---|---|
| 1, 2 | Open-ended research prompt; codebook convention | `code/02_open_ended/data_generation/prompts/research_prompt.txt`, `data/raw/pathways/CODEBOOK.md` |
| 3, 5, 14 to 17 | Fixed-menu immigration prompts | `data/agent_runs/issp_menu_anti/inputs/` |
| 4, 6 to 8 | The same prompts in an earlier cohort (model AD055) | not shipped; checked against pages 14 to 17 |
| 9 | Card-Krueger research prompt | `data/agent_runs/card_luna/packet/research-prompt.txt` |
| 10 | A team brief that no shipped cohort received | none; its text is pinned by a checksum |
| 11, 12 | Commitment and comparison prompts | `data/agent_runs/card_luna/packet/` and the other shipped copies |
| 13 | Immigration analyst request | `data/agent_runs/issp_request_luna/inputs/research_prompt.txt` |
| 18 to 21 | Placebo research prompt and codebook openings | `code/03_placebo/data_generation/runtime/` |
| 22 | Card-Krueger match reviewer's prompt | assembled by `code/09_card_agents/data_generation/terra/rereview_gl.py` from `data/agent_runs/card_luna/review_prompt.json` |

The paper uses pages 1, 2, 9, 11 to 13 and 18 to 22. The other pages are built so that the page numbers stay
those of the paper's source.
