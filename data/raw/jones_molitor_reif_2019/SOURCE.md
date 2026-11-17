# Illinois Workplace Wellness Study, public-use online survey data

Used by module 03 (real-data placebo design, study label `wellness`).

- **Citation.** Jones, Damon, David Molitor, and Julian Reif. 2019. "What Do Workplace Wellness Programs Do?
  Evidence from the Illinois Workplace Wellness Study." *Quarterly Journal of Economics* 134 (4): 1747-1791, doi:10.1093/qje/qjz023.
  Replication data: Harvard Dataverse, doi:10.7910/DVN/VELJKG (version 2.0). The same public-use data are
  also released at Zenodo, doi:10.5281/zenodo.4064859.
- **File obtained.** `illinois-wellness-data-master.zip` from the Dataverse record (MD5
  `10c05e4c515e344cf6fa3ec74140136a`, matching the checksum Dataverse lists; SHA-256
  `054e3c3170aec5e594e6b34874a10fc4e59ed7e40390d74f34ff5aaa44a064cf`). Obtained 6 October 2026;
  license checked against the Dataverse API on 10 November 2026.
- **Files shipped.** `online_surveys.csv`, the archive member
  `illinois-wellness-data-master/data/csv/online_surveys.csv`, unmodified (SHA-256
  `e1f985da88e67432b020550904db8077390d4568220598877fa1a62deec10e6d`); its variable codebook
  `online_surveys.codebook.txt` (archive member `data/codebooks/online_surveys.codebook.txt`, SHA-256
  `8a41dc6c5c77c318ce13f2623c9d0259dfd84818478d3ded4cea16588ec2749a`); and the archive's `LICENSE.md`
  (SHA-256 `a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499`), both also unmodified. The
  rest of the archive is not used.
- **License.** CC0 1.0 (Dataverse record and the repository's `LICENSE.md`). Redistribution is allowed.
- **Content.** One row per employee who completed the baseline survey (4,834 rows), with survey answers in
  2016, 2017 and 2018, the public strata variables (sex, age group, race), the real treatment indicator and
  `hra_c_yr1`, which records whether a treated employee completed the 2016 health risk assessment (empty for
  the 1,534 control employees). The public file has no employee identifiers. Employee class and salary
  quartile, the other two strata variables of the original randomization, are not public.
- **Use.** `code/03_placebo/01_build_base_files.R` keeps 25 survey items at all three waves and 6 items
  at the two follow-ups under new names, the three strata variables and a block index, and drops the real
  treatment and `hra_c_yr1`. The script lists each source column (`<stem>_0716`, `_0717`, `_0718`) beside
  its new name; `online_surveys.codebook.txt` defines the source columns, and
  `code/03_placebo/data_generation/runtime/codebooks/wellness.md` the columns the agents saw.
