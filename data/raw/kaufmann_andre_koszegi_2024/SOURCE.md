# Socially responsible consumers: survey data

Used by module 03 (real-data placebo design, study label `consumers`).

- **Citation.** Kaufmann, Marc, Peter Andre, and Botond Kőszegi. 2024. "Understanding Markets with Socially
  Responsible Consumers." *Quarterly Journal of Economics* 139 (3): 1989-2035, doi:10.1093/qje/qjae009.
  Replication package: Harvard Dataverse, doi:10.7910/DVN/49CETN (version 1.1).
- **Files obtained.** From the Dataverse record, `SRC_replication/out/data_out/wide.dta` in its original
  Stata format (Dataverse file 8941438, MD5 `43dc7740821d4d1e652dcac70554a089` as Dataverse lists it),
  `SRC_replication/codebook/codebook.xlsx` and `SRC_replication/README.md`. Obtained 6 October 2026;
  license checked against the Dataverse API on 10 November 2026.
- **Files shipped, unmodified.** `wide.dta` (SHA-256
  `6b50fd9b57eb7d794896df4d5f4ef4185888b99886a7c5b0368f5a6d471ea959`, 15.3 MB), the authors' variable
  codebook `codebook.xlsx` (SHA-256 `5b1753fe83cf9c759873b8ff8a6d836324fc227bbcce10a4666ff08039e7df9d`; MD5
  `43cca720f4c0b3ce36b30edb991347af` as Dataverse lists it) and their `README.md` (SHA-256
  `15235d3b64c8d4c9da2f152f19a1233a22bed31948a24e89eed5e55fd0d6368a`; MD5
  `d2f23f8404f9493f841fa147e956925d` as Dataverse lists it).
- **License.** The Dataverse record lists CC0 1.0; the authors' README states CC BY-NC 4.0. Under the
  stricter terms, the file may be redistributed with attribution for non-commercial use, which is how this
  package uses it.
- **Content.** The authors' analysis file of the main survey (2,000 respondents, 115 columns), built by their
  cleaning code from the raw survey export. It contains four open-text answers (`explanation`,
  `wtp_explain_same`, `wtp_explain_diff`, `wtp_explain_pos`), a free-text gender field (`gender_4_TEXT`,
  filled for one respondent) and two respondent identifiers (`ResponseId`, `PROLIFIC_PID`). The authors'
  README states that they replaced the Prolific IDs with randomly generated ones. It is shipped as
  published, by the authors' decision.
- **Use.** `code/03_placebo/01_build_base_files.R` keeps 28 columns (demographics, the assigned topics,
  beliefs and willingness to pay) under new names and codes, and none of the identifiers or open-text fields.
  The 29 source columns it reads are listed in that script. `codebook.xlsx` defines 16 of them; the other 13
  are derived by the authors' cleaning code and are defined in the table below.
  `code/03_placebo/data_generation/runtime/codebooks/consumers.md` defines the columns the agents saw. The
  survey had no binary treatment; the placebo design adds an invented one.

Source columns the module reads that `codebook.xlsx` does not define. The definitions follow the authors' cleaning
code (`code/cleaning/clean_survey.R`, `clean_text_beliefs.R` and `merge.R` in their deposit), and each holds for
all 2,000 rows of `wide.dta`.

| Column | Definition |
|---|---|
| `racewhite_yes`, `raceblack_yes`, `racehispanic_yes`, `raceindigenous_yes`, `raceasian_yes` | 1 if the respondent ticked `race_1` (White or Caucasian), `race_2` (Black or African American), `race_3` (Hispanic or Latino), `race_4` (American Indian, Native American or Alaska Native) or `race_5` (Asian), in that order; else 0 |
| `wtp_1_censored`, `wtp_2_censored` | `wtp_1` and `wtp_2` capped at their 95th percentile among the respondents with the same valuation scenario (`conseq`) |
| `conseq_strict` | 1 if `wtp_ratio` is 0 |
| `conseq_weak` | 1 if `wtp_ratio` is below 1 |
| `conseq_deont` | 1 if `wtp_ratio` is 1 |
| `conseq_dontcare` | 1 if `wtp_1` is 0; `wtp_ratio` is then missing and the three indicators above are 0 |
| `code_effect_simple` | The hand-coded category of the open-text answer `explanation`, simplified from `code_effect`: `naive`, `dampening` (which absorbs `tiny`), `spillover` or `other` (which absorbs `junk` and `misunderstanding`) |
| `explanation_dampening` | 1 if `code_effect_simple` is `dampening` |
