# ISSP Role of Government microdata and the country-level indicators

The immigration application (modules 05, 06 and 07) uses one respondent-level analysis file, `issp_clean.csv`
(154,921 respondents, 46 countries, 107 country-waves, 29 columns; SHA-256
`bd2363cec654ea710d9bdbd389f23993c87b6ee7df5bf48a27394b64a4884491`), built from the five ISSP Role of Government
modules and a country-year panel of indicators. Only the indicator panel ships; the five ISSP files must be obtained
from GESIS.

| File | Shipped | Source |
|---|---|---|
| `cri_macro.csv` | yes | Country-year panel of the Crowdsourced Replication Initiative, 46 countries, 1980-2017 (SHA-256 `639fe8d1097b9e6e4d0312cb07ceba3079c21de901bea98870e23d7d468920e0`) |
| `cri_macro_codebook.xlsx` | yes | Its codebook (SHA-256 `23cc6260892d5eda52a47fe803030efe879d3959c356e0f7e060461c6b44844b`) |
| `ZA1490.dta`, `ZA1950.dta`, `ZA2900.dta`, `ZA4700.dta`, `ZA6900_v2-0-0.dta` | no | ISSP Role of Government I-V, GESIS |

## ISSP Role of Government modules

| File used | GESIS study | Data file version and DOI | SHA-256 of the file used |
|---|---|---|---|
| `ZA1490.dta` | ZA1490, Role of Government I, ISSP 1985 | 1.0.0, https://doi.org/10.4232/1.1490 | `fe54f0eea7d8c9ea327f3175e7dc08882929e6d4335e69d43ebdb7b34ebbb80a` |
| `ZA1950.dta` | ZA1950, Role of Government II, ISSP 1990 | 1.0.0, https://doi.org/10.4232/1.1950 | `dae4ba2929759da6c1fecc4abe4f3534f54aec6563b94bbcd35e4d9e24b75e30` |
| `ZA2900.dta` | ZA2900, Role of Government III, ISSP 1996 | 1.0.0, https://doi.org/10.4232/1.2900 | `c26fa85da920f95f84d86ff3e7061dfcdd1508e5d78b24e73f272e15e8885b46` |
| `ZA4700.dta` | ZA4700, Role of Government IV, ISSP 2006 | 1.0.0 (first edition, 18 August 2008), https://doi.org/10.4232/1.4700 | `6a188efd74a817e7fecbf073ca001ded2319eb4d875d23caaa666935156f561a` |
| `ZA6900_v2-0-0.dta` | ZA6900, Role of Government V, ISSP 2016 | 2.0.0, https://doi.org/10.4232/1.13052 | `af39261f95b717e86d2cc4ff9d50a396d741ee58253639cdf903779ff7627ebc` |

The files used are the GESIS releases distributed with the Crowdsourced Replication Initiative's materials (GitHub
`nbreznau/CRI`, commit `dd5dcb27a190bfb27209018f1b02af3fedda9f55`, folder `data/Provided Materials/`). Downloads from
GESIS of the same versions were not tested against these checksums, so the build only warns when an input's checksum
differs; the checksum of the file it builds decides.

Terms: ISSP data are distributed by GESIS free of charge for scientific use after registration, under GESIS's terms
of use, which do not permit passing the data on to third parties. The CRI repository's CC0 label does not license
the ISSP files it contains.

## Country-level indicators

`cri_macro.csv` and `cri_macro_codebook.xlsx` are copied unchanged from the same CRI commit and folder
(https://github.com/nbreznau/CRI/tree/dd5dcb27a190bfb27209018f1b02af3fedda9f55/data/Provided%20Materials). They are
the indicator panel of Breznau, Rinke, Wuttke et al. (2022), PNAS 119(44), e2203150119, compiled from World Bank, UN,
OECD, SWIID and other public sources. The repository is released under CC0-1.0; users should also cite the underlying
sources listed in the codebook.

## Access and rebuild

1. Register with GESIS (https://www.gesis.org/en/issp) and download the Stata files of ZA1490 (v1.0.0), ZA1950
   (v1.0.0), ZA2900 (v1.0.0) and ZA6900 (v2.0.0). GESIS now serves ZA4700 as version 2.0.0, which adds the samples of
   Argentina, Northern Ireland and Slovakia and does not give the file the agents received; version 1.0.0 (the first
   edition of 18 August 2008) is supplied by GESIS on request.
2. Save them in this folder under the names in the table above.
3. From the package root, run (about 3 seconds; Python with pandas and numpy):

       python -I code/05_immigration_personas/data_generation/build_issp_file.py --src data/raw/issp \
           --out data/derived/05_immigration_personas/issp_clean.csv

   The script writes nothing unless the output's SHA-256 is `bd2363ce...4491`, the file the agents received.

## Miao et al.'s distributed file

The Miao et al. cohorts of module 05 ran on `examples/issp/data/data_clean.csv` at commit
`7c8925f9663e34b09e14e9c14ad67a6c3b547bb9` of https://github.com/jmiao24/AgenticBootstrap (152,594 rows; SHA-256
`90e1b6278666d2b8d2ef07758ad93a22d3bf1fe19524c829a2bc54c33e309d13`), a respondent-level file derived from the ISSP,
which is not shipped. In 1985 and 1990 its outcome columns hold other ISSP questions, and in 1990 and 1996 most
country groups carry another country's label.
