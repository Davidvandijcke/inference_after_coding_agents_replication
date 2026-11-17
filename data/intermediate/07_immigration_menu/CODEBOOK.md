# Codebook: data/intermediate/07_immigration_menu

Aggregates computed from the restricted ISSP analysis file (`issp_clean.csv`, see `data/raw/issp/SOURCE.md`). No
respondent-level data. `code/07_immigration_menu/run.sh` reads them; the scripts in
`code/07_immigration_menu/data_generation/` regenerate them into a folder the user names.

The menu is the 108 country-level OLS specifications of `data/agent_runs/issp_menu_anti/inputs/CATALOGUE.json`:
the respondent data are averaged to country-years and then to countries, and the outcome's country mean is regressed
on the immigrant-stock measure (and controls), with HC3 standard errors and a t reference.

## bank_anti.csv.gz and bank_pro.csv.gz

The same 20,000 subsets of 13 of the 46 countries (drawn with `numpy.random.default_rng(2026103021)`), with the
programmatic result of each workflow's rule on each subset. Written by `data_generation/prepare_anti.py` and
`prepare_pro.py` (about one minute each, each as `bank.csv.gz` in the folder the user names), which reproduce their
decompressed content exactly. Their compressed bytes are not reproduced: the gzip header stores a file name
(`bank.csv`) and the time of writing (30 October 2026 in the shipped files).

| Column | File | Meaning |
|---|---|---|
| `bank_index` | both | Subset index, 0 to 19,999 |
| `countries` | anti | ISO 3166 numeric codes of the 13 countries |
| `p_target` | anti | p-value of the target AD061 on the subset (1 where it cannot be fitted) |
| `stratum` | anti | `tail` if `p_target` is at most u = 0.025607, else `body` |
| `p_primary`, `stratum_primary` | pro | The same for AD098 (u = 0.754507), the fixed-model and frontier target |
| `p_second`, `stratum_second` | pro | The same for AD099 (u = 0.918940), the client-pressure target |
| `frontier_pick` | both | Menu model with the most negative (anti) or most positive (pro) HC3 t-statistic on the subset; ties go to the lowest model id |
| `client_branch` | both | `aligned` if the named model (AD061, or AD098 for pro) has the expected sign and p < 0.05 on the subset, else `pressure` |
| `client_pick` | both | The client-pressure rule's report: the named model when aligned, else the most extreme alternative |

## menu_full_data.csv

The 108 specifications fitted on the full file with the agents' helper (`catalogue_helper.py --all`), written by
`data_generation/fit_menu_full_data.py` (seconds): `model_id`, `outcome`, `exposure`, `coverage`, `control_set`
(as in the catalogue), `n_countries`, `slope`, `hc3_se`, `hc3_t` and `raw_hc3_p` (two-sided p-value of the slope,
t with n_countries minus the number of coefficients degrees of freedom).
