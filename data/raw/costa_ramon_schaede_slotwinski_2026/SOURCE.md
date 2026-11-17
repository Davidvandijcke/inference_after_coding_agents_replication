# Financial information and maternal labor supply: derived survey file

Used by module 03 (real-data placebo design, study label `teachers`; the participants are employed mothers,
nearly all of them classroom teachers in a public school system).

- **Citation.** Costa-Ramón, Ana, Michaela Slotwinski, Ursina Schaede, and Anne Ardila Brenøe. 2026. "(Not)
  Thinking About the Future: Financial Information and Maternal Labor Supply." *Quarterly Journal of
  Economics* 141 (2): 1335-1382, doi:10.1093/qje/qjag003. Replication package: Harvard Dataverse,
  doi:10.7910/DVN/A9ZFFF (version 1.1), file `QJE_replicationpackage_transfer.zip` (45.8 MB; MD5
  `f7bd67b87560e601015fbfebe5385d4e` as Dataverse lists it; SHA-256
  `7a500b4d0dd6973d758b73a27a4d17e3fe92df91aaf44f0f43634960a117ac07`). Obtained 6 October 2026; license
  checked against the Dataverse API on 10 November 2026. This folder's name lists three of the four authors.
- **License.** CC0 1.0 (Dataverse record).
- **Source file, not shipped.** The design uses one member of that archive,
  `QJE_replicationpackage_transfer/data/2_clean/1_intermediate/3_clean_fu1.dta` (SHA-256
  `f4e3833abe9224da9b01255139d37f70144798c2aefc46b0787d9a04ed7abc1f`, 3,813 rows, 416 columns). It is not
  redistributed here; the package ships only the numeric extract below, which holds the columns the design
  uses.
- **Files shipped.** `teachers_base.csv` (2,794 rows: the randomization block and 70 survey columns, all
  numeric, under the names the agents saw) and `teachers_design.json` (its row count and the number of
  participants the authors assigned to treatment, 933, which fixes the placebo's treated count). Neither file
  contains a string, identifier, date or treatment column.
- **Regeneration.** Download the archive from the Dataverse record and run
  `Rscript code/03_placebo/data_generation/build_teachers_base.R QJE_replicationpackage_transfer.zip OUT_DIR`.
  It checks both SHA-256 values above, keeps the randomized survey sample with an observed age group, reads
  only the 70 survey columns it lists (source name and new name side by side) and the real assignment
  `treat_all` (to count the treated), and writes the two files, which equal the shipped ones byte for byte.
- **Column meanings.** `code/03_placebo/data_generation/runtime/codebooks/teachers.md`, the codebook the
  agents received, defines every column of `teachers_base.csv`. The real design randomized schools and then
  teachers within treated schools; the placebo assignment is individual-level within age-group blocks.
