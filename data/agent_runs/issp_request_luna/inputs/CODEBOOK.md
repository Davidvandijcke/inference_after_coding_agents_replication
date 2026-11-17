# ISSP Role of Government 1985–2016: respondent file with country indicators

`data.csv` has one row per survey respondent. It combines responses to the International Social Survey Programme
(ISSP) modules on the Role of Government with country-level indicators for the survey year. A country-wave is one
country surveyed in one module. The file has 29 columns. Missing values are empty cells.

## Sources

| Wave | Archive study | Module year |
|---|---|---:|
| `ZA1490` | ISSP 1985 Role of Government I | 1985 |
| `ZA1950` | ISSP 1990 Role of Government II | 1990 |
| `ZA2900` | ISSP 1996 Role of Government III | 1996 |
| `ZA4700` | ISSP 2006 Role of Government IV (first GESIS edition, 2008) | 2006 |
| `ZA6900` | ISSP 2016 Role of Government V (version 2.0.0) | 2016 |

Countries took part in different modules; few took part in all of them.

The survey data are distributed by GESIS (Cologne). Country indicators come from a country-year panel (1980–2017)
compiled from the World Bank World Development Indicators, the UN Population Division, the OECD, the Standardized
World Income Inequality Database (SWIID) and the Quality of Government (QoG) standard dataset. Each respondent
carries the indicator values of their country for the module year (1985, 1990, 1996, 2006 or 2016). National
fieldwork sometimes took place in the year before or after the module year; indicator values are not shifted for
this.

No survey weights are included. Where a country fielded separate regional or group samples in a module, they are
pooled under the national code without reweighting, so such groups are not represented in proportion to their
population share.

## Variables

### Identifiers

| Variable | Description |
|---|---|
| `wave` | ISSP archive study number of the module: `ZA1490`, `ZA1950`, `ZA2900`, `ZA4700`, `ZA6900`. |
| `year` | Module year: 1985, 1990, 1996, 2006, 2016. |
| `iso_country` | Numeric ISO 3166-1 country code. |
| `cntry_wave` | Country-wave identifier, `"<iso_country>_<year>"`. |

### Attitudes to government responsibility (outcomes)

All six items follow the question *"On the whole, do you think it should or should not be the government's
responsibility to ..."*, answered on a four-point scale. All countries in all waves used this four-point scale.

| Variable | Item wording ("... the government's responsibility to ...") | Archive variable 1985 / 1990 / 1996 / 2006 / 2016 |
|---|---|---|
| `jobs` | provide a job for everyone who wants one | V101 / v49 / v36 / V25 / v21 |
| `healthcare` | provide health care for the sick | V103 / v51 / v38 / V27 / v23 |
| `old_age` | provide a decent standard of living for the old | V104 / v52 / v39 / V28 / v24 |
| `unemployed` | provide a decent standard of living for the unemployed | V106 / v54 / v41 / V30 / v26 |
| `income_diff` | reduce income differences between the rich and the poor | V107 / v55 / v42 / V31 / v27 |
| `housing` | provide decent housing for those who can't afford it | not asked in 1985 / v57 / v44 / V33 / v29 |

Coding (0–1, higher = more government responsibility):

| Response | Value |
|---|---:|
| Definitely should be | 1 |
| Probably should be | 2/3 (0.6667) |
| Probably should not be | 1/3 (0.3333) |
| Definitely should not be | 0 |
| Can't choose, no answer, refused, not asked | missing |

`housing` was not asked in 1985. A few items were not asked in individual country-waves; they are missing there.

### Respondent characteristics

| Variable | Description and coding |
|---|---|
| `age` | Age in years (15–99). Missing in country-waves whose archive file records age only in bands. |
| `female` | 1 = female, 0 = male. |
| `employed` | 1 = currently in paid work (full-time, part-time or less than part-time, employees and self-employed alike), 0 = not in paid work (unemployed, in education, retired, homemaker, disabled, other not in the labour force, unpaid helping family member). Source: 1990–2006 current employment status (codes 1–3 = 1, codes 4–10 = 0); 2016 `WORK` (1 = currently in paid work; 2 or 3 = 0). In 1985 the archive distinguishes only employed and unemployed respondents; those outside the labour force share the missing code with item non-response, report no working hours, and are coded 0. |
| `educ` | Highest completed education level, harmonized ordinal scale 0–5 (table below). Available 1996, 2006 and 2016. Missing for 1985 and 1990: the education categories in those archive files are country-specific and unlabelled and cannot be mapped to a common scale. |
| `topbot` | Self-placement on a 10-point top–bottom scale ("In our society there are groups which tend to be towards the top and groups which tend to be towards the bottom"): 1 = bottom, 10 = top. Asked in 2006 and 2016 only, and not in every country-wave. |

Harmonized education (`educ`):

| `educ` | Level | 1996 (`v205`) | 2006 (`DEGREE`) | 2016 (`DEGREE`) |
|---:|---|---|---|---|
| 0 | No formal qualification, or incomplete primary | 1 None (incl. still at school/university), 2 Incomplete primary | 0 No formal qualification, incomplete primary | 0 No formal education |
| 1 | Primary completed (lowest formal qualification) | 3 Primary completed | 1 Lowest formal qualification | 1 Primary school |
| 2 | Lower or incomplete secondary | 4 Incomplete secondary | 2 Above lowest qualification | 2 Lower secondary |
| 3 | Upper secondary completed | 5 Secondary completed | 3 Higher secondary completed | 3 Upper secondary |
| 4 | Post-secondary below a tertiary degree | 6 Semi-higher, incomplete university | 4 Above higher secondary level | 4 Post-secondary, non-tertiary |
| 5 | Tertiary degree completed | 7 University completed | 5 University degree completed | 5 Lower-level tertiary, 6 Upper-level tertiary |

The ISSP education classification changed in 2011 (the 2016 categories follow ISCED 2011), and national
assignments to categories differ across waves. The 2016 category 5 includes short-cycle tertiary programmes as well
as bachelor's degrees, so level 5 is broader in 2016 than in 2006. Comparisons of `educ` within a country across
waves should allow for this.

### Country indicators (values for the module year)

| Variable | Description | Unit | Source |
|---|---|---|---|
| `migstock_wb` | Immigrant stock: foreign-born population (foreign citizens where a country reports only citizenship) as a share of total population. The source publishes the migrant stock every five years; values between are interpolated. Available 1985–2006; none for 2016. | percent of population (0–100) | World Bank WDI SM.POP.TOTL divided by SP.POP.TOTL |
| `migstock_un` | Immigrant stock, same concept, linearly interpolated between the UN benchmark years (1990, 1995, …, 2015, 2017). Available 1990–2016; none for 1985. | percent of population | UN DESA International Migrant Stock 2017, Table 3 |
| `migstock_oecd` | Foreign-born population share, defined consistently across OECD countries. OECD members only, many gaps; none for 1985 and 2016. | percent of population | OECD foreign-born population |
| `migstock` | Combined immigrant stock: `migstock_wb` where available, otherwise `migstock_un`. In this file 2016 values are therefore UN values. | percent of population | derived |
| `mignet_un` | Net migration rate: immigrants minus emigrants per 1,000 inhabitants per year, interpolated from five-year UN estimates. None for 2016. | per 1,000 inhabitants | UN World Population Prospects 2017 |
| `migflow_pct` | Net migration rate expressed as a percentage: `mignet_un / 10`. | percent of population per year | derived |
| `pop_wb` | Total population. | persons | World Bank SP.POP.TOTL |
| `gdp_wb` | GDP per capita. | current US dollars | World Bank NY.GDP.PCAP.CD |
| `log_gdp` | Natural logarithm of `gdp_wb`. | log current US dollars | derived |
| `ginid_solt` | Gini index of disposable (post-tax, post-transfer) household income, mean of the SWIID imputations. | 0–1 | SWIID 7.0 |
| `gini_wb` | Gini index as published by the World Bank (income or consumption basis, varies by country). Sparse. | 0–100 | World Bank SI.POV.GINI |
| `wdi_unempilo` | Unemployment rate, ILO modelled estimate, population 15+. Starts in 1991. | percent of labour force | World Bank WDI via QoG |
| `socx_oecd` | Public social expenditure; five-yearly before 2010 with linear interpolation between data points. OECD members only. | percent of GDP | OECD SOCX |
| `al_ethnic` | Ethnic fractionalization index (constant over time within a country). | 0–1 | Alesina et al. (2003) via QoG |

Indicator gaps: some country-waves have no indicators at all or lack individual indicators (for example, no
immigrant-stock value or no GDP value); `wdi_unempilo` is missing for all 1985 and 1990 country-waves; `ginid_solt`
is incomplete in 2016; `socx_oecd` is missing for non-OECD countries and for some OECD country-waves.

## Notes for use

- Rows are distinct respondents. Because the file keeps only 29 variables, some distinct respondents have identical
  values in every column (mostly in 1985–1996, where age, education or self-placement are unavailable for some
  samples). They are separate interviews, not repeated records.
- Item non-response differs across items and country-waves.
- Country indicators are constant within a country-wave.
