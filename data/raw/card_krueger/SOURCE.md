# Card and Krueger (1994): New Jersey and Pennsylvania fast-food survey

Citation: Card, David, and Alan B. Krueger. 1994. "Minimum Wages and Employment: A Case Study of the Fast-Food
Industry in New Jersey and Pennsylvania." American Economic Review 84 (4): 772-793.

Source: the data set posted by David Card, "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in
New Jersey and Pennsylvania," at https://davidcard.berkeley.edu/data_sets.html, file `njmin.zip`. The six files here
are the archive's contents, unmodified: `public.dat` (the 410 stores, both survey waves, a fixed-width text file
with "." for missing values), `codebook` (column positions and coding), `read.me`, `check.sas` (a SAS program that
reads the data and makes some of the paper's tabulations), and `survey1.nj`, `survey2.nj` (the two questionnaires).
The files are taken from the authors' local copy of the archive; the original download date is not recorded.

| File | Bytes | SHA-256 |
|---|---|---|
| `public.dat` | 82,410 | `04bde0cad5540980f32ce099c6dad369e2f05494698071d8a65b3e1cbe9ca53a` |
| `codebook` | 3,822 | `1d801d7bdfa77ad2138d79ead54eef830913a2bdf4861a50ee013b3474e4b52f` |
| `read.me` | 1,247 | `1e368163606f6bfcd6d037bc91c4703ced8a53c0d2341ac678f35c11ee14d02b` |
| `check.sas` | 3,387 | `b77fd592a7e6791266272e76d03d0c7213d95bd00014c754aec6141cbd8e48fc` |
| `survey1.nj` | 5,803 | `076743c2d981764171ad4ab8fed12c1c23855f1be59a66ac985c716503a199bc` |
| `survey2.nj` | 5,480 | `ffa7c81804561f6641841e27122ba7140046773747efa71535a866a753173f86` |

Terms: the files are posted publicly, without registration, and the source page states no license for them. Its
footer carries the notice "©2014 The Regents of the University of California" (page read on 11 November 2026). By
the authors' decision the files are redistributed here unmodified, with attribution to Card and Krueger (1994) and
to the source page, so that the package holds the data the agents analyzed. Replicators can instead download
`njmin.zip` from the source page and compare the hashes above.

Use in this package: the file the agents received, `data/agent_runs/card_luna/packet/data.csv`, is `public.dat` with
the codebook's variable names in lower case, missing entries written as `NA`, and a column `store_id` (state and
sheet number joined by a space, since sheet numbers repeat across the two states). `code/08_card/01_original.R`
builds it from these files and checks that it is byte-identical to the agents' copy.
